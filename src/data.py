"""Load original file-order pairs and create an EDA view."""

from pathlib import Path
import pandas as pd


def load_training_pairs(data_dir):
    data_dir = Path(data_dir)
    labels = pd.read_csv(data_dir / "train.csv")
    rows = []
    for _, row in labels.iterrows():
        article_id = int(row["id"])
        folder = data_dir / "train" / f"article_{article_id:04d}"
        rows.append({
            "article_id": article_id,
            "text_1": (folder / "file_1.txt").read_text(encoding="utf-8"),
            "text_2": (folder / "file_2.txt").read_text(encoding="utf-8"),
            "real_text_id": int(row["real_text_id"])
        })
    return pd.DataFrame(rows)


def load_test_pairs(data_dir):
    rows = []
    for folder in sorted((Path(data_dir) / "test").glob("article_*")):
        if folder.is_dir():
            rows.append({
                "article_id": int(folder.name.split("_")[1]),
                "text_1": (folder / "file_1.txt").read_text(encoding="utf-8"),
                "text_2": (folder / "file_2.txt").read_text(encoding="utf-8")
            })
    return pd.DataFrame(rows)


def make_eda_view(pairs):
    """Use known labels for analysis only; retain all original pairs."""
    rows = []
    for _, row in pairs.iterrows():
        real_id = int(row["real_text_id"])
        fake_id = 2 if real_id == 1 else 1
        rows.append({
            "article_id": int(row["article_id"]),
            "real_text": row[f"text_{real_id}"],
            "fake_text": row[f"text_{fake_id}"],
            "real_text_id": real_id
        })
    return pd.DataFrame(rows)
