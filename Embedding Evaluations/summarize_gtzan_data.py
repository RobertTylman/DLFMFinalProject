from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "dl4m-evaluation-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GTZAN embedding data layout.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Embedding root containing Data/genres_original and Data/genres_augmented.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_overview"),
        help="Directory for overview CSVs and plots.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write CSV tables in addition to graph images.",
    )
    return parser.parse_args()


def original_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((root / "Data" / "genres_original").glob("*/*.npy")):
        rows.append(
            {
                "split": "original",
                "genre": path.parent.name,
                "track_id": path.stem,
                "degradation_type": "clean",
                "degradation_value": 0,
                "path": path.relative_to(root).as_posix(),
            }
        )
    return rows


def augmented_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((root / "Data" / "genres_augmented").glob("*/*dB/*/*.npy")):
        relative = path.relative_to(root)
        _, _, degradation_type, value_name, genre, _ = relative.parts
        rows.append(
            {
                "split": "augmented",
                "genre": genre,
                "track_id": path.stem,
                "degradation_type": degradation_type,
                "degradation_value": int(value_name.removesuffix("dB")),
                "path": relative.as_posix(),
            }
        )
    return rows


def plot_bar(series: pd.Series, output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", ax=ax, color="#386cb0")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_noise_grid(counts: pd.DataFrame, output_path: Path) -> None:
    pivot = counts.pivot_table(
        index="degradation_type",
        columns="degradation_value",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(pivot.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)), labels=[f"{value} dB" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[name.replace("_", " ").title() for name in pivot.index])
    ax.set_title("Augmented Files By Noise Type And SNR")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, int(pivot.iat[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Files")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_genre_noise_counts(counts: pd.DataFrame, output_path: Path) -> None:
    pivot = counts.pivot_table(
        index="genre",
        columns="degradation_type",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Augmented Files Per Genre")
    ax.set_xlabel("")
    ax.set_ylabel("Files")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Noise Type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = original_rows(root) + augmented_rows(root)
    data = pd.DataFrame(rows)
    split_counts = data.groupby("split", as_index=False).size().rename(columns={"size": "count"})
    original_genre_counts = (
        data[data["split"] == "original"].groupby("genre").size().rename("count").sort_index()
    )
    augmented_counts = (
        data[data["split"] == "augmented"]
        .groupby(["degradation_type", "degradation_value", "genre"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    if args.write_csv:
        data.to_csv(output_dir / "all_files.csv", index=False)
        split_counts.to_csv(output_dir / "split_counts.csv", index=False)
        original_genre_counts.to_csv(output_dir / "original_genre_counts.csv")
        augmented_counts.to_csv(output_dir / "augmented_counts_by_noise_db_genre.csv", index=False)

    plot_bar(
        original_genre_counts,
        output_dir / "original_genre_counts.png",
        "Clean Original Files Per Genre",
        "Files",
    )
    plot_noise_grid(augmented_counts, output_dir / "augmented_noise_db_counts.png")
    plot_genre_noise_counts(augmented_counts, output_dir / "augmented_genre_noise_counts.png")

    print("Genres:")
    print(", ".join(original_genre_counts.index.tolist()))
    print("\nSplit counts:")
    print(split_counts.to_string(index=False))
    print(f"\nWrote {output_dir.resolve()}")


if __name__ == "__main__":
    main()
