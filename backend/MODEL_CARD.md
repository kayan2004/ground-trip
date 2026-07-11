# Model Card: Smart Travel Planner

This consolidates the ML/AI components scattered across `backend/README.md` into one reference:
what each one is, what data it was fit/evaluated on, real measured numbers (not estimates), current
status, and honest known limitations. Each section links to the fuller write-up in
`backend/README.md` for implementation detail - this file is the "what and why," not a duplicate of
the "how."

No formal bias/fairness audit has been performed on any component below. The destination corpus
skews toward well-documented, tourism-heavy destinations - a function of what Wikivoyage/
OpenTripMap/Numbeo actually cover - which likely underrepresents less-touristed regions. None of
these components are exposed to end users directly; every one sits behind the LangGraph agent
pipeline, which degrades gracefully (`status="partial"`) on any single component's failure rather
than surfacing a raw model error.

The cosine recommender (component 1) is the essential ranking criterion - it runs on every request
and is what "recommendation" means in this app. The LightGBM ranker (component 2) is the sole
additional trained ML model, and it's secondary by design: it only re-orders what cosine already
retrieved rather than replacing it, and only runs when both `RANKER_ENABLED=true` (the default)
and its trained artifact are present, falling back to pure cosine order automatically otherwise.
There used to be
a third model here - an SVC travel-style classifier - but it was fully removed from the repo
2026-07-11 (code, trained artifact, and training CSV all deleted; see git history at
`git log --diff-filter=D -- backend/app/services/classifier.py` for what it looked like) after
being dormant and unreachable since 2026-07-05. It's gone, not just retired - don't expect to find
`artifacts/ml/` or `app/services/classifier.py` in this checkout.

## 1. Destination recommender (structured pre-filter + pgvector cosine re-rank) — production, essential

- **Purpose:** the primary recommendation mechanism the live agent actually uses, and the only one
  every request goes through - every other component below is optional or offline.
- **Model:** not a trained model - a SQL pre-filter (budget ceiling with null-passthrough, region,
  tag-weight threshold) followed by a pgvector cosine similarity re-rank over Voyage embeddings
  (`voyage-4-lite`, 1024-dim) via the `<=>` operator.
- **"Training" data:** none to train - relies on the real 219-destination corpus's real embeddings
  (`app/services/destination_ingestion.py`).
- **Evaluation** (`scripts/evaluate_recommendations.py`, 18 hand-labeled queries, real corpus,
  2026-07-08): **recall@3 = 0.34, recall@5 = 0.44, recall@10 = 0.66, MRR = 0.75, NDCG@5 = 0.49**.
  Investigated the lowest-scoring query directly rather than assume a defect: it returned
  semantically reasonable results that simply weren't on the hand-picked expected list - a
  limitation of binary ground truth on open-ended queries, not a recommender defect (see
  `backend/README.md`'s "Recommendation quality eval" section for the full finding).
- **Status:** production, always active on every request.

## 2. LightGBM learning-to-rank ranker — secondary, the only other ML model in play

- **Purpose:** re-orders the cosine-retrieved candidate slate before truncation - never the
  primary signal. Cosine similarity remains the essential criterion (component 1): the ranker only
  reorders what cosine already retrieved, and only runs at all when `RANKER_ENABLED=true` **and**
  a trained model exists at `artifacts/ranker/model.joblib` - it degrades to pure cosine order
  automatically if either condition isn't met, rather than blocking a request on it.
- **Model:** `LGBMRanker`, `lambdarank` objective, 4 features (`cosine_sim`, `tag_match_count`,
  `budget_delta`, `region_match`).
- **Training data:** **COLD-START PRIOR** - 150 synthetic queries (real destination embeddings
  perturbed with calibrated Gaussian noise, since no live `VOYAGE_API_KEY` was available when this
  was built), labeled by a documented heuristic with added noise. **Not real user feedback** - the
  `feedback` table is empty; the real-feedback retrain path (`scripts/train_ranker.py retrain`)
  exists but has never fired for real data.
- **Evaluation** (`artifacts/ranker/model_metadata.json`): NDCG@3 = 0.916, NDCG@5 = 0.929 - against
  its **own** synthetic bootstrap heuristic, not independent ground truth, so this measures fit to
  the heuristic, not real recommendation quality. `cosine_sim` dominates feature importance
  (~13x the next-highest feature) - in practice this keeps the ranker's output close to cosine
  order even when it's active.
- **Status:** `RANKER_ENABLED=true` by default (`app/core/config.py`) - active on every request as
  long as the trained artifact is present, which it is in this repo. Off only if explicitly set to
  `false` or the artifact is removed.
- **Known limitation:** the label formula is itself a function of the same 4 features the model
  trains on, so it mostly approximates that formula rather than learning genuine preference; in
  real traffic `tag_match_count` is near-constant (0) since the live graph node never populates
  `required_tags`. Being on by default despite training on a synthetic cold-start prior (not real
  feedback) is a deliberate tradeoff, not an oversight - see `backend/README.md`'s
  "Learning-to-Rank" section.

## 3. HDBSCAN + UMAP destination clustering — offline, tags applied

- **Purpose:** weighted travel-style tags per destination (`destinations.tags`), human-readable
  cluster names (`tag_definitions`).
- **Model:** UMAP (cosine metric, dimensionality reduction) → HDBSCAN (density-based soft
  clustering, `all_points_membership_vectors` for weighted multi-tag assignment) on the same
  Voyage embeddings as the recommender.
- **Training data:** the full 219-destination corpus, unsupervised (no labels).
- **Evaluation** (`artifacts/clustering/quality_report.json`, `stability_report.json`): 5 clusters,
  4.1% noise (9/219 destinations), silhouette (UMAP space) = 0.527, DBCV = 0.439, stability mean
  pairwise ARI = 0.874 across 5 UMAP seeds (min 0.730, comfortably above the 0.7 instability
  threshold - not flagged unstable).
- **Status:** all 3 phases (cluster → name → apply-tags) run for real against the live corpus
  (2026-07-06). `tag_definitions` has 5 real named clusters: "South American Cultural Heritage",
  "European Architectural Heritage", "Asian Cultural Heritage", "Oceania Cultural Heritage",
  "Dynamic Urban Metropolises". `destinations.tags` holds real tag-name keys, not the raw
  `cluster_id` placeholders it held before.
- **Known limitation:** produces the tags but nothing in the live graph node sets
  `required_tags` from them yet - the recommender's tag-threshold filter (component 1) exists and
  is correct, but is currently unexercised in production traffic.

## 4. LLM usage (Gemini, third-party) — production

- **Purpose:** structured field extraction from user prompts, trip-plan synthesis, offline cluster
  naming (component 3's naming phase).
- **Model:** currently `gemini-3.1-flash-lite` (paid, not the free Gemma tier), pinned as the
  single model for all 3 call sites - no fast/strong tiers (removed 2026-07-06; see
  `backend/README.md`'s "Provider-Agnostic LLM Layer" section for why). Model name is a hardcoded
  default in `app/core/config.py`, not an `.env` var (2026-07-09) - a cost/quality decision meant
  to go through code review, not a silent runtime toggle.
- Not a model this project trained - a third-party model, accessed through a provider-agnostic
  abstraction (`app/services/llm_providers/`); Anthropic is kept configured but dormant behind the
  same interface.
- **Known limitation:** the free-tier fallback (`gemma-4-26b-a4b-it`, used if billing isn't set up)
  spends a real, sometimes-large token budget on internal "thinking" before the visible answer (up
  to ~1500 tokens observed on a trivial prompt) - `max_tokens` is configured generously (4096
  default) to avoid truncating mid-thought regardless of which model is active.
- **Status:** production default, real live calls verified working. Confirmed live (2026-07-09) in
  a direct comparison against `gemini-3.1-pro-preview` on the same prompt: `flash-lite` was ~5x
  faster and ~9x cheaper (~$0.001/run at this app's prompt sizes) with equivalent answer quality
  (see `app/services/llm_providers/usage_logging.py`'s pricing table, verified live against each
  provider's current pricing page).
