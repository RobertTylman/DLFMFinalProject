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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
            "Train a genre classifier on clean GTZAN embeddings and evaluate it "
            "on noisy/augmented GTZAN embeddings."
        )
    )
    parser.add_argument(
        "--embedding-root",
        action="append",
        type=Path,
        required=True,
        help="Root folder containing Data/genres_original and Data/genres_augmented.",
    )
    parser.add_argument(
        "--model-label",
        action="append",
        help=(
            "Label for the corresponding --embedding-root. If omitted, the checkpoint "
            "stem from embedding_manifest.json is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/genre_classification"),
        help="Directory for classification tables and plots.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional path for per-file predictions. Only written with --write-csv.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write CSV tables in addition to graph images.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed for the classifier.",
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
    if not checkpoint:
        return None

    return Path(checkpoint).stem


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
        raise ValueError(f"Expected a dB degradation folder, got {name!r}")
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

    if not original_root.exists():
        raise FileNotFoundError(f"Missing clean original embeddings: {original_root}")
    if not augmented_root.exists():
        raise FileNotFoundError(f"Missing augmented embeddings: {augmented_root}")

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


def train_classifier(
    train_items: list[EmbeddingItem],
    c_value: float,
    seed: int,
) -> tuple[object, list[str]]:
    x_train = load_matrix(train_items)
    y_train = [item.genre for item in train_items]

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )
    classifier.fit(x_train, y_train)
    return classifier, sorted(set(y_train))


def evaluate_model(root: Path, model: str, c_value: float, seed: int) -> pd.DataFrame:
    train_items, test_items = list_data(root)
    classifier, _ = train_classifier(train_items, c_value=c_value, seed=seed)
    x_test = load_matrix(test_items)
    predictions = classifier.predict(x_test)

    probabilities = classifier.predict_proba(x_test)
    classes = classifier.classes_
    max_probabilities = probabilities.max(axis=1)

    rows = []
    for item, predicted, confidence in zip(test_items, predictions, max_probabilities):
        true_probability = probabilities[
            len(rows),
            int(np.flatnonzero(classes == item.genre)[0]),
        ]
        rows.append(
            {
                "model": model,
                "track_id": item.track_id,
                "degradation_type": item.degradation_type,
                "degradation_value": item.degradation_value,
                "true_genre": item.genre,
                "predicted_genre": predicted,
                "correct": int(predicted == item.genre),
                "prediction_confidence": float(confidence),
                "true_genre_probability": float(true_probability),
                "query_path": item.relative_path,
            }
        )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = dict(
        num_queries=("correct", "count"),
        accuracy=("correct", "mean"),
        mean_prediction_confidence=("prediction_confidence", "mean"),
        mean_true_genre_probability=("true_genre_probability", "mean"),
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
    by_genre = (
        results.groupby(["model", "true_genre"], as_index=False)
        .agg(**metrics)
        .sort_values(["model", "true_genre"])
    )

    confusion = (
        results.groupby(["model", "true_genre", "predicted_genre"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = (
        confusion.groupby(["model", "true_genre"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "true_genre_total"})
    )
    confusion = confusion.merge(totals, on=["model", "true_genre"])
    confusion["rate"] = confusion["count"] / confusion["true_genre_total"]

    return {
        "by_snr_genre": by_snr_genre,
        "by_noise_snr_genre": by_noise_snr_genre,
        "by_snr": by_snr,
        "by_noise_snr": by_noise_snr,
        "by_genre": by_genre,
        "confusion": confusion,
    }


def percent_label(value: float) -> str:
    return f"{value * 100:.0f}%"


def plot_per_genre_by_snr(by_snr_genre: pd.DataFrame, output_dir: Path) -> None:
    colors = {20: "#2ca02c", 10: "#ff7f0e", 0: "#d62728"}
    snr_order = [20, 10, 0]

    for model in by_snr_genre["model"].drop_duplicates():
        data = by_snr_genre[by_snr_genre["model"] == model]
        pivot = (
            data.pivot(index="true_genre", columns="degradation_value", values="accuracy")
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

        ax.set_title(f"Per-Genre Accuracy Vs GTZAN Folder Genre By SNR\n{model}")
        ax.set_xlabel("Top-1 genre accuracy (%) - averaged across noise types")
        ax.set_ylabel("Ground-truth GTZAN genre")
        ax.set_yticks(y, labels=pivot.index)
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(title="SNR", loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"per_genre_accuracy_by_snr_{safe_label(model)}.png", dpi=180)
        plt.close(fig)


def condition_label(noise_type: str, snr: int) -> str:
    names = {
        "crowd_noise": "Crowd",
        "street_noise": "Street",
        "white_noise": "White",
    }
    return f"{names.get(noise_type, noise_type.replace('_', ' ').title())}\n{snr} dB"


def plot_noise_snr_heatmap(by_noise_snr_genre: pd.DataFrame, output_path: Path) -> None:
    data = by_noise_snr_genre.copy()
    data["condition"] = [
        condition_label(row.degradation_type, row.degradation_value)
        for row in data.itertuples(index=False)
    ]
    condition_order = (
        data[["degradation_type", "degradation_value", "condition"]]
        .drop_duplicates()
        .sort_values(["degradation_type", "degradation_value"])
    )
    conditions = condition_order["condition"].tolist()
    genres = sorted(data["true_genre"].drop_duplicates())
    models = data["model"].drop_duplicates().tolist()

    fig, axes = plt.subplots(len(models), 1, figsize=(11, 4.2 * len(models)), squeeze=False)
    image = None
    for ax, model in zip(axes[:, 0], models):
        pivot = (
            data[data["model"] == model]
            .pivot(index="true_genre", columns="condition", values="accuracy")
            .reindex(index=genres, columns=conditions)
        )
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
        ax.set_aspect("auto")
        ax.set_title(model)
        ax.set_yticks(range(len(genres)), labels=genres)
        ax.set_xticks(range(len(conditions)), labels=conditions)
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iat[row, col]
                text_color = "black" if value >= 0.65 else "white"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.suptitle("Genre Accuracy Vs GTZAN Folder Genre By Noise Type And SNR", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.16, right=0.86, top=0.90, bottom=0.08, hspace=0.50)
    if image is not None:
        colorbar_axis = fig.add_axes([0.89, 0.16, 0.025, 0.68])
        fig.colorbar(image, cax=colorbar_axis, label="Accuracy")
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_overall_by_snr(by_snr: pd.DataFrame, output_path: Path) -> None:
    pivot = by_snr.pivot(index="degradation_value", columns="model", values="accuracy").sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.mul(100).plot(kind="bar", ax=ax)
    ax.set_title("Overall Genre Accuracy Vs GTZAN Folder Genre By SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Embedding Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(confusion: pd.DataFrame, output_path: Path) -> None:
    genres = sorted(set(confusion["true_genre"]) | set(confusion["predicted_genre"]))
    models = confusion["model"].drop_duplicates().tolist()

    fig, axes = plt.subplots(len(models), 1, figsize=(9.5, 4.7 * len(models)), squeeze=False)
    image = None
    for ax, model in zip(axes[:, 0], models):
        pivot = (
            confusion[confusion["model"] == model]
            .pivot(index="true_genre", columns="predicted_genre", values="rate")
            .reindex(index=genres, columns=genres)
            .fillna(0)
        )
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="magma")
        ax.set_aspect("auto")
        ax.set_title(model)
        ax.set_xlabel("Predicted genre")
        ax.set_ylabel("Ground-truth GTZAN genre")
        ax.set_xticks(range(len(genres)), labels=genres)
        ax.set_yticks(range(len(genres)), labels=genres)
        ax.tick_params(axis="x", rotation=35)
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iat[row, col]
                if value >= 0.05:
                    text_color = "black" if value >= 0.65 else "white"
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.suptitle("Genre Classification Confusion Matrix Vs GTZAN Folder Genre", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.14, right=0.86, top=0.90, bottom=0.10, hspace=0.55)
    if image is not None:
        colorbar_axis = fig.add_axes([0.89, 0.18, 0.025, 0.64])
        fig.colorbar(image, cax=colorbar_axis, label="Rate Within True Genre")
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_outputs(
    results: pd.DataFrame,
    output_dir: Path,
    results_csv: Path | None,
    write_csv: bool,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = summarize(results)

    if write_csv:
        if results_csv is not None:
            results.to_csv(results_csv, index=False)
        tables["by_snr_genre"].to_csv(output_dir / "accuracy_by_snr_genre.csv", index=False)
        tables["by_noise_snr_genre"].to_csv(output_dir / "accuracy_by_noise_snr_genre.csv", index=False)
        tables["by_snr"].to_csv(output_dir / "accuracy_by_snr.csv", index=False)
        tables["by_noise_snr"].to_csv(output_dir / "accuracy_by_noise_snr.csv", index=False)
        tables["by_genre"].to_csv(output_dir / "accuracy_by_genre.csv", index=False)
        tables["confusion"].to_csv(output_dir / "confusion_matrix.csv", index=False)

    plot_per_genre_by_snr(tables["by_snr_genre"], output_dir)
    plot_noise_snr_heatmap(tables["by_noise_snr_genre"], output_dir / "accuracy_by_noise_snr_genre.png")
    plot_overall_by_snr(tables["by_snr"], output_dir / "overall_accuracy_by_snr.png")
    plot_confusion(tables["confusion"], output_dir / "confusion_matrices.png")

    return tables


def main() -> None:
    args = parse_args()
    roots = [root.expanduser().resolve() for root in args.embedding_root]
    labels = model_labels(roots, args.model_label)
    output_dir = args.output_dir
    results_csv = args.results_csv or output_dir / "classification_results.csv"

    frames = []
    for root, label_value in zip(roots, labels):
        print(f"Training clean-original genre classifier for {label_value}")
        print(f"  root: {root}")
        frame = evaluate_model(root, label_value, c_value=args.c, seed=args.seed)
        print(f"  noisy test files: {len(frame):,}")
        print(f"  accuracy: {frame['correct'].mean() * 100:.1f}%")
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    tables = write_outputs(results, output_dir, results_csv, write_csv=args.write_csv)

    print("\nOverall accuracy by SNR:")
    print(
        tables["by_snr"]
        .pivot(index="degradation_value", columns="model", values="accuracy")
        .mul(100)
        .round(1)
        .to_string()
    )
    if args.write_csv:
        print(f"\nWrote {results_csv.resolve()}")
    print(f"Wrote {output_dir.resolve()}")


if __name__ == "__main__":
    main()
