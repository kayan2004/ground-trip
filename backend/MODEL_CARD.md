# Model Card: Smart Travel Assistant

This consolidates the ML/AI components scattered across `backend/README.md` into one reference:
what each one is, what data it was fit/evaluated on, real measured numbers (not estimates), current
status, and honest known limitations. Each section links to the fuller write-up in
`backend/README.md` for implementation detail - this file is the "what and why," not a duplicate of
the "how."

No formal bias/fairness audit has been performed on any component below. The destination corpus and
the classifier's training data both skew toward well-documented, tourism-heavy destinations - a
function of what Wikivoyage/OpenTripMap/Numbeo actually cover - which likely underrepresents
less-touristed regions. None of these components are exposed to end users directly; every one sits
behind the LangGraph agent pipeline, which degrades gracefully (`status="partial"`) on any single
component's failure rather than surfacing a raw model error.

## 1. Travel-style classifier (SVC) — dormant

- **Purpose (originally):** predict one of 6 travel-style labels from structured destination
  features, gating which recommendations to show.
- **Model:** scikit-learn `SVC` (RBF kernel, `class_weight="balanced"`), selected over Logistic
  Regression and Random Forest by 5-fold stratified cross-validation on macro F1. Trained in
  `backend/notebook/ml.ipynb`.
- **Training data:** `data/travel_destinations_labeled.csv` - 200 rows, hand-labeled, balanced
  across 6 classes (Adventure/Relaxation/Culture/Budget/Luxury/Family, 33-34 rows each). A curated
  assignment artifact, not scraped or crowd-sourced.
- **Evaluation** (`artifacts/ml/model_metadata.json`, `classification_report.json`): 5-fold CV
  macro F1 = **0.965** (std 0.037), accuracy = **0.965** (std 0.037). Per-class F1 ranges 0.94
  (Adventure) to 0.99 (Culture, Luxury).
- **Status:** retired from the live recommendation path (2026-07-05), replaced by the structured
  pre-filter + cosine recommender below. Still fully functional standalone at
  `POST /tools/classify-travel-style` - the artifact and code were kept, not deleted.
- **Why it was replaced, honestly:** the near-perfect CV score is a small-dataset artifact, not
  evidence the classifier generalizes - 200 hand-labeled rows can't capture the real
  219-destination corpus's diversity, and a fixed 6-class taxonomy can't express the
  multi-dimensional travel-style signal that weighted embeddings + soft clustering (component 4,
  below) now provide instead.

## 2. Destination recommender (structured pre-filter + pgvector cosine re-rank) — production

- **Purpose:** the primary recommendation mechanism the live agent actually uses.
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

## 3. LightGBM learning-to-rank ranker — optional, off by default

- **Purpose:** optionally re-orders the cosine-retrieved candidate slate before truncation.
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
  (~13x the next-highest feature).
- **Status:** `RANKER_ENABLED=false` by default.
- **Known limitation:** the label formula is itself a function of the same 4 features the model
  trains on, so it mostly approximates that formula rather than learning genuine preference; in
  real traffic `tag_match_count` is near-constant (0) since the live graph node never populates
  `required_tags`.

## 4. HDBSCAN + UMAP destination clustering — offline, tags applied

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
  `required_tags` from them yet - the recommender's tag-threshold filter (component 2) exists and
  is correct, but is currently unexercised in production traffic.

## 5. LLM usage (Gemini/Gemma, third-party) — production

- **Purpose:** structured field extraction from user prompts, trip-plan synthesis, offline cluster
  naming (component 4's naming phase).
- **Model:** currently Gemma 4 (`gemma-4-26b-a4b-it`), free tier, pinned as the single model for
  all 3 call sites - no fast/strong tiers (removed 2026-07-06; see `backend/README.md`'s
  "Provider-Agnostic LLM Layer" section for why).
- Not a model this project trained - a third-party model, accessed through a provider-agnostic
  abstraction (`app/services/llm_providers/`); Anthropic is kept configured but dormant behind the
  same interface.
- **Known limitation:** Gemma 4 spends a real, sometimes-large token budget on internal "thinking"
  before the visible answer (up to ~1500 tokens observed on a trivial prompt) - `max_tokens` is
  configured generously (4096 default) to avoid truncating mid-thought.
- **Status:** production default, real live calls verified working (2026-07-06 - superseding an
  earlier "unverified" state in this repo's history). Gemini-branded models (`gemini-3.1-*`) are
  available via a one-line config change once billing is set up - they cost real money per token
  (see `app/services/llm_providers/usage_logging.py`'s pricing table, verified live against each
  provider's current pricing page).
