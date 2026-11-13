"""Prepare a bounded, binary-classification sample from data/ml-data.zip."""
from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "ml-data.zip"
TRAIN_PATH = DATA_DIR / "train_mosaic.csv.zip"
TEST_PATH = DATA_DIR / "test_mosaic.csv.zip"
MANIFEST_PATH = DATA_DIR / "ml-data_manifest.json"

RANDOM_SEED = 42
ROWS_PER_SOURCE_FILE = 100_000
SAMPLES_PER_SOURCE_CLASS = 2_500
MAX_ROWS_PER_BINARY_CLASS = 15_000
TEST_SIZE = 0.2

LABEL_COLUMN = "Label"
OUTPUT_LABEL_COLUMN = "label"
IDENTIFIER_COLUMNS = {
    "unnamed: 0",
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
    "similarhttp",
    "simillarhttp",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def has_valid_local_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    """Check the local ZIP signature before asking zipfile to open a member."""
    archive.fp.seek(info.header_offset)
    return archive.fp.read(4) == b"PK\x03\x04"


def read_source_sample(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, source_index: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read a bounded prefix, normalize schema, and sample each binary class."""
    with archive.open(info) as member:
        raw = pd.read_csv(member, nrows=ROWS_PER_SOURCE_FILE, low_memory=False)

    raw.columns = [str(column).strip() for column in raw.columns]
    if raw.columns.has_duplicates:
        raise ValueError("Column names collide after whitespace normalization.")

    label_column = next(
        (column for column in raw.columns if column.casefold() == LABEL_COLUMN.casefold()),
        None,
    )
    if label_column is None:
        raise ValueError(f"Missing expected target column {LABEL_COLUMN!r}.")

    original_labels = raw[label_column].astype("string").str.strip()
    valid_labels = original_labels.notna() & original_labels.ne("")
    binary_labels = original_labels.str.upper().eq("BENIGN").map(
        {True: "BENIGN", False: "MALICIOUS"}
    )

    excluded = {name.casefold() for name in IDENTIFIER_COLUMNS}
    feature_columns = [
        column
        for column in raw.columns
        if column != label_column and column.casefold() not in excluded
    ]
    if not feature_columns:
        raise ValueError("No usable feature columns remain after removing identifiers.")

    features = pd.DataFrame(index=raw.index)
    for column in feature_columns:
        features[column] = pd.to_numeric(raw[column], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    prepared = features.loc[valid_labels].copy()
    prepared[OUTPUT_LABEL_COLUMN] = binary_labels.loc[valid_labels].to_numpy()
    sampled = []
    class_counts: dict[str, int] = {}
    for class_index, class_name in enumerate(("BENIGN", "MALICIOUS")):
        class_rows = prepared.loc[prepared[OUTPUT_LABEL_COLUMN] == class_name]
        class_counts[class_name] = len(class_rows)
        if len(class_rows) > SAMPLES_PER_SOURCE_CLASS:
            class_rows = class_rows.sample(
                n=SAMPLES_PER_SOURCE_CLASS,
                random_state=RANDOM_SEED + source_index * 2 + class_index,
            )
        if not class_rows.empty:
            sampled.append(class_rows)

    return pd.concat(sampled, ignore_index=True), class_counts


def write_compressed_csv(data: pd.DataFrame, destination: Path, csv_name: str) -> Path:
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    data.to_csv(
        temporary_path,
        index=False,
        compression={"method": "zip", "archive_name": csv_name},
    )
    return temporary_path


def main() -> None:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Required input archive was not found: {ARCHIVE_PATH}")

    source_samples = []
    included_sources = []
    skipped_sources = []
    source_label_counts = {}

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        csv_members = [
            info for info in archive.infolist() if info.filename.lower().endswith(".csv")
        ]
        if not csv_members:
            raise ValueError(f"No CSV members were found in {ARCHIVE_PATH}.")

        for source_index, info in enumerate(csv_members):
            if not has_valid_local_header(archive, info):
                logging.warning("Skipping member with invalid ZIP header: %s", info.filename)
                skipped_sources.append({"member": info.filename, "reason": "invalid ZIP local header"})
                continue

            try:
                sample, class_counts = read_source_sample(archive, info, source_index)
            except (OSError, ValueError, zipfile.BadZipFile, pd.errors.ParserError) as error:
                logging.warning("Skipping unreadable member %s: %s", info.filename, error)
                skipped_sources.append({"member": info.filename, "reason": str(error)})
                continue

            if sample.empty:
                logging.warning("Skipping member with no labeled rows: %s", info.filename)
                skipped_sources.append({"member": info.filename, "reason": "no labeled rows"})
                continue

            included_sources.append(info.filename)
            source_label_counts[info.filename] = class_counts
            source_samples.append(sample)
            logging.info(
                "Sampled %s: %s BENIGN, %s MALICIOUS (from at most %s rows)",
                info.filename,
                class_counts["BENIGN"],
                class_counts["MALICIOUS"],
                ROWS_PER_SOURCE_FILE,
            )

    if not source_samples:
        raise ValueError("No readable, labeled CSV members were found in ml-data.zip.")

    combined = pd.concat(source_samples, ignore_index=True)
    counts = combined[OUTPUT_LABEL_COLUMN].value_counts()
    if len(counts) < 2:
        raise ValueError(f"Both BENIGN and MALICIOUS rows are required; found {counts.to_dict()}.")

    per_class_count = min(int(counts.min()), MAX_ROWS_PER_BINARY_CLASS)
    if per_class_count < 10:
        raise ValueError(f"Too few samples per class to prepare train/test data: {counts.to_dict()}.")

    balanced = pd.concat(
        [
            combined.loc[combined[OUTPUT_LABEL_COLUMN] == class_name].sample(
                n=per_class_count,
                random_state=RANDOM_SEED + class_index,
            )
            for class_index, class_name in enumerate(("BENIGN", "MALICIOUS"))
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    labels = balanced[OUTPUT_LABEL_COLUMN]
    features = balanced.drop(columns=[OUTPUT_LABEL_COLUMN])
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_data = train_features.copy()
    train_data[OUTPUT_LABEL_COLUMN] = train_labels.to_numpy()
    test_data = test_features.copy()
    test_data[OUTPUT_LABEL_COLUMN] = test_labels.to_numpy()

    train_tmp = write_compressed_csv(train_data, TRAIN_PATH, "train_mosaic.csv")
    test_tmp = write_compressed_csv(test_data, TEST_PATH, "test_mosaic.csv")
    try:
        train_tmp.replace(TRAIN_PATH)
        test_tmp.replace(TEST_PATH)
    finally:
        train_tmp.unlink(missing_ok=True)
        test_tmp.unlink(missing_ok=True)

    manifest = {
        "source_archive": ARCHIVE_PATH.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": {
            "rows_read_per_csv_member_max": ROWS_PER_SOURCE_FILE,
            "samples_per_source_file_and_class_max": SAMPLES_PER_SOURCE_CLASS,
            "balanced_rows_per_class_max": MAX_ROWS_PER_BINARY_CLASS,
            "test_fraction": TEST_SIZE,
            "random_seed": RANDOM_SEED,
        },
        "label_mapping": {"BENIGN": "BENIGN", "all other non-empty labels": "MALICIOUS"},
        "removed_identifier_columns": sorted(IDENTIFIER_COLUMNS),
        "features": list(train_features.columns),
        "included_members": included_sources,
        "skipped_members": skipped_sources,
        "source_label_counts_in_read_prefixes": source_label_counts,
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "train_class_counts": train_labels.value_counts().to_dict(),
        "test_class_counts": test_labels.value_counts().to_dict(),
        "note": (
            "This is a reproducible bounded sample, not a full scan of the 31 GB uncompressed archive. "
            "The original ml-data.zip was not modified."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Prepared exclusively from %s", ARCHIVE_PATH.name)
    logging.info("Train rows: %s; class counts: %s", len(train_data), train_labels.value_counts().to_dict())
    logging.info("Test rows: %s; class counts: %s", len(test_data), test_labels.value_counts().to_dict())
    logging.info("Feature columns: %s", len(train_features.columns))
    logging.info("Wrote %s, %s, and %s", TRAIN_PATH.name, TEST_PATH.name, MANIFEST_PATH.name)


if __name__ == "__main__":
    main()
