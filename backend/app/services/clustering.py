"""Offline HDBSCAN soft-clustering of the destinations corpus.

Business logic for scripts/cluster_destinations.py. Never imported by the
request-time graph - this runs offline, once per corpus change, exactly
like scripts/ingest_destinations.py -> app/services/destination_ingestion.py.
"""

import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hdbscan
import httpx
import joblib
import numpy as np
import umap
from hdbscan import validity as hdbscan_validity
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models.destination import Destination
from app.db.models.tag_definition import TagDefinition
from app.services.llm import propose_cluster_tag

MEMBERSHIP_ARTIFACT_FILENAME = "cluster_members.json"
RAW_MEMBERSHIP_NPZ_FILENAME = "membership_vectors.npz"
QUALITY_REPORT_FILENAME = "quality_report.json"
STABILITY_REPORT_FILENAME = "stability_report.json"
UMAP_REDUCER_FILENAME = "umap_reducer.joblib"
HDBSCAN_CLUSTERER_FILENAME = "hdbscan_clusterer.joblib"
UMAP_SCATTER_FILENAME = "umap_scatter.png"
MEMBERSHIP_HISTOGRAM_FILENAME = "membership_weight_histogram.png"
NAMING_PROMPTS_DIRNAME = "naming_prompts"

# Below this mean pairwise ARI across stability re-runs, the clustering is
# flagged as sensitive to UMAP's random initialization rather than reflecting
# real corpus structure. Not a hard scientific threshold - a documented,
# defensible heuristic (see backend/README.md).
STABILITY_ARI_WARNING_THRESHOLD = 0.7


@dataclass(slots=True)
class DestinationVector:
    id: str
    name: str
    country: str
    region: str | None
    budget_level: str | None
    embedding: np.ndarray


@dataclass(slots=True)
class ClusteringRun:
    vectors: list[DestinationVector]
    umap_embedding: np.ndarray
    labels: np.ndarray
    membership: np.ndarray
    n_clusters: int
    reducer: umap.UMAP
    clusterer: hdbscan.HDBSCAN


class DegenerateClusteringError(RuntimeError):
    """Raised when HDBSCAN found fewer than 1 real cluster."""


async def load_embedded_destination_vectors(session: AsyncSession) -> list[DestinationVector]:
    result = await session.execute(
        select(
            Destination.id,
            Destination.name,
            Destination.country,
            Destination.region,
            Destination.budget_level,
            Destination.embedding,
        ).where(Destination.embedding.is_not(None))
    )
    vectors: list[DestinationVector] = []
    for dest_id, name, country, region, budget_level, embedding in result.all():
        vectors.append(
            DestinationVector(
                id=str(dest_id),
                name=name,
                country=country,
                region=region,
                budget_level=budget_level,
                embedding=np.asarray(embedding, dtype=np.float64),
            )
        )
    return vectors


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against a degenerate zero embedding
    return matrix / norms


def fit_umap(
    matrix: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> umap.UMAP:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    reducer.fit(matrix)
    return reducer


def fit_hdbscan(
    umap_embedding: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> hdbscan.HDBSCAN:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        gen_min_span_tree=True,
        # HDBSCAN has no random_state of its own; the approximate minimum
        # spanning tree algorithm is the only source of run-to-run
        # nondeterminism it has. Disabling it makes the main clustering run
        # fully reproducible for a fixed UMAP embedding - all variation in
        # the stability check (see run_stability_check) then comes
        # exclusively from UMAP's random_state, which is the honest signal
        # we actually want to measure.
        approx_min_span_tree=False,
    )
    clusterer.fit(umap_embedding)
    return clusterer


def run_clustering(
    vectors: list[DestinationVector],
    *,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    min_cluster_size: int,
    min_samples: int | None,
    random_state: int,
) -> ClusteringRun:
    raw_matrix = np.vstack([v.embedding for v in vectors])
    normalized = l2_normalize(raw_matrix)

    reducer = fit_umap(
        normalized,
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    umap_embedding = np.asarray(reducer.embedding_)

    clusterer = fit_hdbscan(
        umap_embedding,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.labels_
    n_clusters = int(len(set(labels.tolist()) - {-1}))

    if n_clusters < 1:
        raise DegenerateClusteringError(
            "HDBSCAN found 0 clusters (everything is noise) with "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}. "
            "Try a smaller --min-cluster-size."
        )

    membership = hdbscan.all_points_membership_vectors(clusterer)
    # all_points_membership_vectors returns a 1D array when there is exactly
    # one cluster; normalize to the (n, n_clusters) shape used everywhere
    # else so downstream indexing doesn't need a special case.
    membership = np.asarray(membership)
    if membership.ndim == 1:
        membership = membership.reshape(-1, 1)

    return ClusteringRun(
        vectors=vectors,
        umap_embedding=umap_embedding,
        labels=labels,
        membership=membership,
        n_clusters=n_clusters,
        reducer=reducer,
        clusterer=clusterer,
    )


def threshold_membership_to_tags(
    membership_row: np.ndarray, *, threshold: float
) -> dict[str, float]:
    return {
        str(cluster_id): float(weight)
        for cluster_id, weight in enumerate(membership_row)
        if weight > threshold
    }


def compute_quality_metrics(run: ClusteringRun, *, membership_threshold: float) -> dict[str, Any]:
    labels = run.labels
    n_total = len(labels)
    noise_mask = labels == -1
    noise_ratio = float(noise_mask.sum()) / n_total if n_total else 0.0

    hard_cluster_sizes = {
        str(cluster_id): int((labels == cluster_id).sum()) for cluster_id in range(run.n_clusters)
    }
    soft_cluster_sizes = {
        str(cluster_id): int((run.membership[:, cluster_id] > membership_threshold).sum())
        for cluster_id in range(run.n_clusters)
    }

    silhouette = _safe_silhouette(run.umap_embedding, labels)
    dbcv = _safe_dbcv(run.umap_embedding, labels)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_destinations": n_total,
        "n_clusters": run.n_clusters,
        "noise_ratio": noise_ratio,
        "noise_count": int(noise_mask.sum()),
        "hard_cluster_sizes": hard_cluster_sizes,
        "soft_cluster_sizes": soft_cluster_sizes,
        "silhouette_umap_space": silhouette,
        "dbcv": dbcv,
        "membership_threshold": membership_threshold,
    }


def _safe_silhouette(umap_embedding: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels != -1
    unique_labels = set(labels[mask].tolist())
    if len(unique_labels) < 2:
        return None
    try:
        return float(silhouette_score(umap_embedding[mask], labels[mask]))
    except ValueError:
        return None


def _safe_dbcv(umap_embedding: np.ndarray, labels: np.ndarray) -> float | None:
    if len(set(labels.tolist()) - {-1}) < 2:
        return None
    try:
        return float(
            hdbscan_validity.validity_index(
                umap_embedding.astype(np.float64), labels, metric="euclidean"
            )
        )
    except Exception:  # noqa: BLE001 - DBCV is a nice-to-have diagnostic; a
        # pathological cluster geometry (e.g. a cluster with near-duplicate
        # points) must degrade to "unavailable", not abort the whole report.
        return None


def run_stability_check(
    vectors: list[DestinationVector],
    *,
    base_random_state: int,
    n_runs: int,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    min_cluster_size: int,
    min_samples: int | None,
) -> dict[str, Any]:
    """Re-fit the full UMAP -> HDBSCAN pipeline across n_runs distinct UMAP
    random_state seeds and report pairwise Adjusted Rand Index between the
    resulting label assignments (noise included as its own class).

    HDBSCAN itself is deterministic here (approx_min_span_tree=False), so
    every bit of run-to-run variation is attributable to UMAP's random
    initialization - the actual thing we want to measure the sensitivity of.
    """
    seeds = [base_random_state + offset for offset in range(n_runs)]
    label_sets: list[np.ndarray] = []
    cluster_counts: list[int] = []

    for seed in seeds:
        try:
            run = run_clustering(
                vectors,
                umap_n_components=umap_n_components,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                random_state=seed,
            )
            label_sets.append(run.labels)
            cluster_counts.append(run.n_clusters)
        except DegenerateClusteringError:
            label_sets.append(np.full(len(vectors), -1))
            cluster_counts.append(0)

    pairwise_ari = [
        float(adjusted_rand_score(a, b)) for a, b in itertools.combinations(label_sets, 2)
    ]
    mean_ari = float(np.mean(pairwise_ari)) if pairwise_ari else None
    min_ari = float(np.min(pairwise_ari)) if pairwise_ari else None

    return {
        "seeds": seeds,
        "n_runs": n_runs,
        "cluster_counts_per_run": cluster_counts,
        "pairwise_ari": pairwise_ari,
        "mean_ari": mean_ari,
        "min_ari": min_ari,
        "flagged_unstable": mean_ari is not None and mean_ari < STABILITY_ARI_WARNING_THRESHOLD,
        "instability_threshold": STABILITY_ARI_WARNING_THRESHOLD,
    }


def build_membership_dump(run: ClusteringRun, *, membership_threshold: float) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = {str(i): [] for i in range(run.n_clusters)}
    for vector, membership_row in zip(run.vectors, run.membership, strict=True):
        for cluster_id in range(run.n_clusters):
            weight = float(membership_row[cluster_id])
            if weight <= 0.0:
                continue
            clusters[str(cluster_id)].append(
                {
                    "id": vector.id,
                    "name": vector.name,
                    "country": vector.country,
                    "region": vector.region,
                    "budget_level": vector.budget_level,
                    "membership": weight,
                }
            )

    for members in clusters.values():
        members.sort(key=lambda entry: -entry["membership"])

    return {
        "membership_threshold": membership_threshold,
        "n_clusters": run.n_clusters,
        "clusters": clusters,
    }


def save_clustering_artifacts(
    run: ClusteringRun,
    *,
    membership_threshold: float,
    quality_metrics: dict[str, Any],
    stability_report: dict[str, Any] | None,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(run.reducer, artifacts_dir / UMAP_REDUCER_FILENAME)
    joblib.dump(run.clusterer, artifacts_dir / HDBSCAN_CLUSTERER_FILENAME)

    np.savez(
        artifacts_dir / RAW_MEMBERSHIP_NPZ_FILENAME,
        destination_ids=np.array([v.id for v in run.vectors]),
        membership=run.membership,
        labels=run.labels,
    )

    membership_dump = build_membership_dump(run, membership_threshold=membership_threshold)
    (artifacts_dir / MEMBERSHIP_ARTIFACT_FILENAME).write_text(
        json.dumps(membership_dump, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (artifacts_dir / QUALITY_REPORT_FILENAME).write_text(
        json.dumps(quality_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if stability_report is not None:
        (artifacts_dir / STABILITY_REPORT_FILENAME).write_text(
            json.dumps(stability_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    _save_umap_scatter(run, artifacts_dir / UMAP_SCATTER_FILENAME)
    _save_membership_histogram(run, artifacts_dir / MEMBERSHIP_HISTOGRAM_FILENAME)


def _save_umap_scatter(run: ClusteringRun, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A dedicated 2D UMAP fit for visualization, distinct from the
    # n_components-dim embedding actually clustered on - plotting the first
    # two dimensions of a >2D UMAP embedding is not a faithful 2D layout.
    normalized = l2_normalize(np.vstack([v.embedding for v in run.vectors]))
    scatter_reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=run.reducer.random_state,
    )
    scatter_embedding = scatter_reducer.fit_transform(normalized)

    fig, ax = plt.subplots(figsize=(9, 7))
    noise_mask = run.labels == -1
    ax.scatter(
        scatter_embedding[noise_mask, 0],
        scatter_embedding[noise_mask, 1],
        c="lightgray",
        s=12,
        label="noise",
        alpha=0.6,
    )
    non_noise = ~noise_mask
    scatter = ax.scatter(
        scatter_embedding[non_noise, 0],
        scatter_embedding[non_noise, 1],
        c=run.labels[non_noise],
        cmap="tab20",
        s=16,
    )
    ax.set_title(f"Destinations UMAP (2D) colored by HDBSCAN cluster (n={run.n_clusters})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    legend1 = ax.legend(*scatter.legend_elements(), title="cluster", loc="upper right", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_membership_histogram(run: ClusteringRun, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonzero_weights = run.membership[run.membership > 0.0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nonzero_weights, bins=40, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of nonzero soft-cluster membership weights")
    ax.set_xlabel("membership weight")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


async def write_cluster_id_tags_to_db(
    session: AsyncSession,
    run: ClusteringRun,
    *,
    membership_threshold: float,
) -> None:
    for vector, membership_row in zip(run.vectors, run.membership, strict=True):
        tags = threshold_membership_to_tags(membership_row, threshold=membership_threshold)
        await session.execute(
            update(Destination).where(Destination.id == vector.id).values(tags=tags)
        )
    await session.commit()


@dataclass(slots=True)
class ClusterNamingResult:
    cluster_id: int
    tag_name: str
    description: str
    quality_metrics: dict[str, Any]


async def name_clusters(
    session: AsyncSession,
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    membership_dump: dict[str, Any],
    quality_report: dict[str, Any],
    top_n: int,
    artifacts_dir: Path,
) -> list[ClusterNamingResult]:
    results: list[ClusterNamingResult] = []
    prompts_dir = artifacts_dir / NAMING_PROMPTS_DIRNAME
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id_str, members in membership_dump["clusters"].items():
        cluster_id = int(cluster_id_str)
        top_members = members[:top_n]
        member_ids = [entry["id"] for entry in top_members]
        poi_kinds_by_id = await _load_poi_kind_summaries(session, member_ids)

        example_destinations = [
            {
                "name": entry["name"],
                "country": entry["country"],
                "region": entry["region"],
                "budget_level": entry["budget_level"],
                "membership": entry["membership"],
                "poi_kinds": poi_kinds_by_id.get(entry["id"], "unknown"),
            }
            for entry in top_members
        ]

        cluster_quality_metrics = {
            "cluster_id": cluster_id,
            "hard_cluster_size": quality_report["hard_cluster_sizes"].get(cluster_id_str),
            "soft_cluster_size": quality_report["soft_cluster_sizes"].get(cluster_id_str),
            "global_silhouette_umap_space": quality_report["silhouette_umap_space"],
            "global_dbcv": quality_report["dbcv"],
            "global_noise_ratio": quality_report["noise_ratio"],
        }

        proposal = await propose_cluster_tag(
            http_client,
            settings,
            cluster_id=cluster_id,
            example_destinations=example_destinations,
            quality_metrics=cluster_quality_metrics,
        )

        (prompts_dir / f"cluster_{cluster_id}.json").write_text(
            json.dumps(
                {
                    "cluster_id": cluster_id,
                    "example_destinations": example_destinations,
                    "quality_metrics": cluster_quality_metrics,
                    "proposal": proposal.model_dump(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        await _upsert_tag_definition(
            session,
            cluster_id=cluster_id,
            tag_name=proposal.tag_name,
            description=proposal.description,
            quality_metrics=cluster_quality_metrics,
        )

        results.append(
            ClusterNamingResult(
                cluster_id=cluster_id,
                tag_name=proposal.tag_name,
                description=proposal.description,
                quality_metrics=cluster_quality_metrics,
            )
        )

    await session.commit()
    return results


async def _load_poi_kind_summaries(
    session: AsyncSession, destination_ids: list[str]
) -> dict[str, str]:
    if not destination_ids:
        return {}

    result = await session.execute(
        select(Destination.id, Destination.raw_sources["opentripmap_kind_counts"]).where(
            Destination.id.in_(destination_ids)
        )
    )
    summaries: dict[str, str] = {}
    for dest_id, kind_counts in result.all():
        if not kind_counts:
            summaries[str(dest_id)] = "none recorded"
            continue
        top_kinds = sorted(kind_counts.items(), key=lambda item: -item[1])[:5]
        summaries[str(dest_id)] = ", ".join(f"{kind} ({count})" for kind, count in top_kinds)
    return summaries


async def _upsert_tag_definition(
    session: AsyncSession,
    *,
    cluster_id: int,
    tag_name: str,
    description: str,
    quality_metrics: dict[str, Any],
) -> None:
    statement = pg_insert(TagDefinition).values(
        cluster_id=cluster_id,
        tag_name=tag_name,
        description=description,
        quality_metrics=quality_metrics,
    )
    statement = statement.on_conflict_do_update(
        index_elements=[TagDefinition.cluster_id],
        set_={
            "tag_name": statement.excluded.tag_name,
            "description": statement.excluded.description,
            "quality_metrics": statement.excluded.quality_metrics,
        },
    )
    await session.execute(statement)


async def apply_approved_tag_names(
    session: AsyncSession,
    *,
    membership_dump: dict[str, Any],
) -> dict[str, Any]:
    """Rewrite destinations.tags from cluster_id keys to tag_name keys.

    Source of truth is the persisted membership dump (not the current
    destinations.tags column - see module docstring) plus whatever
    tag_definitions currently holds. Re-runnable at any point: clusters
    without an approved tag_definitions row are simply omitted from the
    written tags rather than blocking the whole run.
    """
    tag_names_by_cluster_id = await _load_tag_names_by_cluster_id(session)
    threshold = membership_dump["membership_threshold"]

    per_destination_tags: dict[str, dict[str, float]] = {}
    for cluster_id_str, members in membership_dump["clusters"].items():
        cluster_id = int(cluster_id_str)
        tag_name = tag_names_by_cluster_id.get(cluster_id)
        if tag_name is None:
            continue
        for entry in members:
            if entry["membership"] <= threshold:
                continue
            per_destination_tags.setdefault(entry["id"], {})[tag_name] = entry["membership"]

    all_destination_ids = {
        entry["id"] for members in membership_dump["clusters"].values() for entry in members
    }
    for destination_id in all_destination_ids:
        tags = per_destination_tags.get(destination_id, {})
        await session.execute(
            update(Destination).where(Destination.id == destination_id).values(tags=tags)
        )
    await session.commit()

    current_cluster_ids = {int(cid) for cid in membership_dump["clusters"]}
    clusters_applied = sorted(current_cluster_ids & set(tag_names_by_cluster_id))
    clusters_missing_names = sorted(current_cluster_ids - set(tag_names_by_cluster_id))
    return {
        "destinations_updated": len(all_destination_ids),
        "clusters_applied": clusters_applied,
        "clusters_missing_names": clusters_missing_names,
    }


async def _load_tag_names_by_cluster_id(session: AsyncSession) -> dict[int, str]:
    result = await session.execute(select(TagDefinition.cluster_id, TagDefinition.tag_name))
    return {cluster_id: tag_name for cluster_id, tag_name in result.all()}


def load_membership_dump(artifacts_dir: Path) -> dict[str, Any]:
    path = artifacts_dir / MEMBERSHIP_ARTIFACT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `cluster_destinations.py cluster` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_quality_report(artifacts_dir: Path) -> dict[str, Any]:
    path = artifacts_dir / QUALITY_REPORT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `cluster_destinations.py cluster` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
