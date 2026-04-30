from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "dl4m-evaluation-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingItem:
    path: Path
    relative_path: str
    track_id: str
    genre: str
    degradation_type: str
    degradation_value: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact-track retrieval: noisy GTZAN embeddings query a clean "
            "GTZAN original embedding index."
        )
    )
    parser.add_argument("--embedding-root", action="append", type=Path, required=True)
    parser.add_argument("--model-label", action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("results/exact_song_retrieval"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write CSV tables in addition to graph images.",
    )
    return parser.parse_args()


def checkpoint_label(root: Path) -> str | None:
    manifest_path = root / "embedding_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint = manifest.get("checkpoint")
    return Path(checkpoint).stem if checkpoint else None


def safe_label(value: str) -> str:
    label = re.sub(r"\s+", "_", value.strip())
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return label.strip("_") or "model"


def model_labels(roots: list[Path], provided: list[str] | None) -> list[str]:
    if provided is not None and len(provided) != len(roots):
        raise ValueError("--model-label must be supplied once for each --embedding-root")

    labels = []
    for index, root in enumerate(roots):
        raw = provided[index] if provided is not None else checkpoint_label(root)
        labels.append((raw or root.name).strip() or root.name)

    if len(set(labels)) != len(labels):
        raise ValueError(f"Model labels must be unique, got {labels}")
    return labels


def numeric_db_value(name: str) -> int:
    match = re.fullmatch(r"(-?\d+)dB", name)
    if not match:
        raise ValueError(f"Expected a dB folder, got {name!r}")
    return int(match.group(1))


def original_item(path: Path, root: Path) -> EmbeddingItem:
    return EmbeddingItem(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        track_id=path.stem,
        genre=path.parent.name,
        degradation_type="clean",
        degradation_value=0,
    )


def augmented_item(path: Path, root: Path) -> EmbeddingItem:
    relative = path.relative_to(root)
    # Data/genres_augmented/{noise_type}/{value}dB/{genre}/{track}.npy
    parts = relative.parts
    if len(parts) < 6 or parts[1] != "genres_augmented":
        raise ValueError(f"Unexpected augmented embedding path: {relative.as_posix()}")

    return EmbeddingItem(
        path=path,
        relative_path=relative.as_posix(),
        track_id=path.stem,
        genre=path.parent.name,
        degradation_type=parts[2],
        degradation_value=numeric_db_value(parts[3]),
    )


def list_data(root: Path) -> tuple[list[EmbeddingItem], list[EmbeddingItem]]:
    original_root = root / "Data" / "genres_original"
    augmented_root = root / "Data" / "genres_augmented"
    originals = [original_item(path, root) for path in sorted(original_root.glob("*/*.npy"))]
    augmented = [augmented_item(path, root) for path in sorted(augmented_root.glob("*/*dB/*/*.npy"))]

    if not originals:
        raise FileNotFoundError(f"No clean original .npy files found under {original_root}")
    if not augmented:
        raise FileNotFoundError(f"No augmented .npy files found under {augmented_root}")

    return originals, augmented


def load_embedding(path: Path) -> np.ndarray:
    embedding = np.load(path)
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Embedding has invalid norm: {path}")
    return embedding / norm


def load_matrix(items: list[EmbeddingItem]) -> np.ndarray:
    return np.vstack([load_embedding(item.path) for item in items]).astype(np.float32)


def evaluate_model(root: Path, model: str, batch_size: int) -> pd.DataFrame:
    index_items, query_items = list_data(root)
    index_matrix = load_matrix(index_items)
    query_matrix = load_matrix(query_items)

    index_track_ids = np.array([item.track_id for item in index_items])
    index_genres = np.array([item.genre for item in index_items])
    index_paths = np.array([item.relative_path for item in index_items])

    rows = []
    for start in range(0, len(query_items), batch_size):
        end = min(start + batch_size, len(query_items))
        similarities = query_matrix[start:end] @ index_matrix.T
        top_k = min(5, similarities.shape[1])
        top_indices = np.argpartition(-similarities, kth=top_k - 1, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_indices = np.take_along_axis(top_indices, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)

        for offset, query in enumerate(query_items[start:end]):
            candidate_indices = top_indices[offset]
            candidate_scores = top_scores[offset]
            top1_index = int(candidate_indices[0])
            top1_track = str(index_track_ids[top1_index])
            top1_genre = str(index_genres[top1_index])

            rows.append(
                {
                    "model": model,
                    "track_id": query.track_id,
                    "degradation_type": query.degradation_type,
                    "degradation_value": query.degradation_value,
                    "true_genre": query.genre,
                    "matched_track_id": top1_track,
                    "predicted_genre": top1_genre,
                    "top_1_correct": int(top1_track == query.track_id),
                    "top_5_correct": int(query.track_id in set(index_track_ids[candidate_indices])),
                    "genre_correct": int(top1_genre == query.genre),
                    "cosine_similarity": float(candidate_scores[0]),
                    "query_path": query.relative_path,
                    "matched_path": str(index_paths[top1_index]),
                }
            )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = results.copy()
    results["wrong_song_right_genre"] = (
        (results["top_1_correct"] == 0) & (results["genre_correct"] == 1)
    ).astype(int)

    metrics = dict(
        num_queries=("track_id", "count"),
        top_1_accuracy=("top_1_correct", "mean"),
        top_5_accuracy=("top_5_correct", "mean"),
        retrieved_genre_accuracy=("genre_correct", "mean"),
        wrong_song_right_genre_rate=("wrong_song_right_genre", "mean"),
        mean_cosine_similarity=("cosine_similarity", "mean"),
    )

    by_snr_genre = (
        results.groupby(["model", "true_genre", "degradation_value"], as_index=False)
        .agg(**metrics)
        .sort_values(["model", "true_genre", "degradation_value"])
    )
    by_noise_snr_genre = (
        results.groupby(
            ["model", "degradation_type", "degradation_value", "true_genre"],
            as_index=False,
        )
        .agg(**metrics)
        .sort_values(["model", "degradation_type", "degradation_value", "true_genre"])
    )
    by_snr = (
        results.groupby(["model", "degradation_value"], as_index=False)
        .agg(**metrics)
        .sort_values(["model", "degradation_value"])
    )
    by_noise_snr = (
        results.groupby(["model", "degradation_type", "degradation_value"], as_index=False)
        .agg(**metrics)
        .sort_values(["model", "degradation_type", "degradation_value"])
    )

    return {
        "by_snr_genre": by_snr_genre,
        "by_noise_snr_genre": by_noise_snr_genre,
        "by_snr": by_snr,
        "by_noise_snr": by_noise_snr,
    }


def percent_label(value: float) -> str:
    return f"{value * 100:.0f}%"


def plot_per_genre_by_snr(by_snr_genre: pd.DataFrame, output_dir: Path) -> None:
    colors = {20: "#2ca02c", 10: "#ff7f0e", 0: "#d62728"}
    snr_order = [20, 10, 0]

    for model in by_snr_genre["model"].drop_duplicates():
        data = by_snr_genre[by_snr_genre["model"] == model]
        pivot = (
            data.pivot(index="true_genre", columns="degradation_value", values="top_1_accuracy")
            .reindex(columns=snr_order)
        )
        pivot["average"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("average", ascending=True).drop(columns="average")

        y = np.arange(len(pivot))
        bar_height = 0.24
        offsets = {20: bar_height, 10: 0.0, 0: -bar_height}

        fig, ax = plt.subplots(figsize=(11, 7))
        for snr in snr_order:
            values = pivot[snr].to_numpy()
            bars = ax.barh(
                y + offsets[snr],
                values * 100,
                height=bar_height,
                color=colors[snr],
                label=f"{snr} dB",
            )
            for bar, value in zip(bars, values):
                ax.text(
                    min(value * 100 + 0.8, 101),
                    bar.get_y() + bar.get_height() / 2,
                    percent_label(value),
                    va="center",
                    ha="left",
                    fontsize=9,
                )

        ax.set_title(f"Exact Clean-Track Top-1 Retrieval Accuracy By Genre And SNR\n{model}")
        ax.set_xlabel("Top-1 same-track retrieval accuracy (%) - averaged across noise types")
        ax.set_ylabel("Ground-truth GTZAN genre")
        ax.set_yticks(y, labels=pivot.index)
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(title="SNR", loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"exact_top1_by_genre_snr_{safe_label(model)}.png", dpi=180)
        plt.close(fig)


def plot_overall_by_snr(by_snr: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = by_snr.pivot(index="degradation_value", columns="model", values="top_1_accuracy")
    pivot.sort_index().mul(100).plot(kind="bar", ax=ax)
    ax.set_title("Overall Exact Clean-Track Top-1 Retrieval Accuracy By SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Embedding Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top1_top5(by_snr: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in by_snr["model"].drop_duplicates():
        data = by_snr[by_snr["model"] == model].sort_values("degradation_value")
        ax.plot(data["degradation_value"], data["top_1_accuracy"] * 100, marker="o", label=f"{model} Top-1")
        ax.plot(data["degradation_value"], data["top_5_accuracy"] * 100, marker="x", linestyle="--", label=f"{model} Top-5")

    ax.set_title("Exact Clean-Track Top-1 And Top-5 Retrieval Accuracy By SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(results: pd.DataFrame, output_dir: Path, write_csv: bool) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = summarize(results)

    if write_csv:
        results.to_csv(output_dir / "retrieval_results.csv", index=False)
        tables["by_snr_genre"].to_csv(output_dir / "exact_retrieval_by_snr_genre.csv", index=False)
        tables["by_noise_snr_genre"].to_csv(output_dir / "exact_retrieval_by_noise_snr_genre.csv", index=False)
        tables["by_snr"].to_csv(output_dir / "exact_retrieval_by_snr.csv", index=False)
        tables["by_noise_snr"].to_csv(output_dir / "exact_retrieval_by_noise_snr.csv", index=False)

    plot_per_genre_by_snr(tables["by_snr_genre"], output_dir)
    plot_overall_by_snr(tables["by_snr"], output_dir / "overall_exact_top1_by_snr.png")
    plot_top1_top5(tables["by_snr"], output_dir / "overall_exact_top1_top5_by_snr.png")

    return tables


def main() -> None:
    args = parse_args()
    roots = [root.expanduser().resolve() for root in args.embedding_root]
    labels = model_labels(roots, args.model_label)

    frames = []
    for root, label_value in zip(roots, labels):
        print(f"Evaluating exact-song retrieval for {label_value}")
        frame = evaluate_model(root, label_value, batch_size=args.batch_size)
        print(f"  noisy queries: {len(frame):,}")
        print(f"  top-1 exact-song accuracy: {frame['top_1_correct'].mean() * 100:.1f}%")
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    tables = write_outputs(results, args.output_dir, write_csv=args.write_csv)

    print("\nExact-song Top-1 accuracy by SNR:")
    print(
        tables["by_snr"]
        .pivot(index="degradation_value", columns="model", values="top_1_accuracy")
        .mul(100)
        .round(1)
        .to_string()
    )
    print(f"\nWrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
