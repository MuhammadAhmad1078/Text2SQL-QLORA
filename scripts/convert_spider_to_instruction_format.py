#!/usr/bin/env python3
"""
Convert raw Spider dataset into instruction-tuning format for QLoRA training.

This script reads Spider's JSON files (train_spider.json, dev.json, tables.json)
and produces instruction-format JSON files ready for SFTTrainer.

Spider dataset structure (expected under --spider_dir):
    spider/
    ├── train_spider.json      # Training examples (question + SQL pairs)
    ├── dev.json               # Dev/validation examples (our held-out test set)
    ├── tables.json            # Schema definitions for all databases
    └── database/              # SQLite .db files (used for execution accuracy)
        ├── concert_singer/
        │   └── concert_singer.sqlite
        ├── ...

Output format (one JSON object per example):
    {
        "instruction": "Given the database schema, write a SQL query ...",
        "schema": "CREATE TABLE singer (Singer_ID INTEGER, Name TEXT, ...)",
        "question": "How many singers are from France?",
        "output": "SELECT COUNT(*) FROM singer WHERE Country = 'France'",
        "db_id": "concert_singer"
    }

Design decisions:
    1. Schema is reconstructed as CREATE TABLE statements (not raw column lists)
       because this is closer to what models see in pretraining data — they've
       seen millions of CREATE TABLE statements in code corpora.
    2. We include PRIMARY KEY and FOREIGN KEY constraints in the schema because
       join relationships are critical for multi-table queries. Without them,
       the model has to guess which columns link tables.
    3. db_id is preserved in the output so evaluation scripts can locate the
       correct SQLite database for execution accuracy.
    4. Train/val split is done at the database level (not example level) to
       prevent schema leakage — if we split by example, the model could see
       train examples from a database and then be tested on other examples
       from the same database, inflating metrics.

Usage:
    python scripts/convert_spider_to_instruction_format.py \
        --spider_dir data/spider \
        --output_dir data/processed \
        --val_ratio 0.1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# Schema Reconstruction
# =============================================================================

# Mapping from Spider's type strings to SQL types.
# Spider uses lowercase type names; we normalize to standard SQL types.
SPIDER_TYPE_MAP: dict[str, str] = {
    "text": "TEXT",
    "number": "INTEGER",  # Spider doesn't distinguish INT/FLOAT in types
    "time": "TEXT",        # Times are stored as text in Spider's SQLite DBs
    "boolean": "INTEGER",  # SQLite has no native boolean
    "others": "TEXT",
}


def build_create_table_statements(table_schema: dict[str, Any]) -> str:
    """Reconstruct CREATE TABLE SQL from Spider's schema metadata.

    Spider stores schema info as parallel arrays:
        - table_names_original: ["stadium", "singer", ...]
        - column_names_original: [[-1, "*"], [0, "Stadium_ID"], [0, "Location"], ...]
            where the first element is the table index (-1 = special '*' column)
        - column_types: ["text", "number", ...] (aligned with column_names_original)
        - primary_keys: [1, 8, ...]  (indices into column_names_original)
        - foreign_keys: [[18, 1], [20, 15], ...]  (pairs of column indices)

    This function reconstructs proper CREATE TABLE statements with:
        - Column names and types
        - PRIMARY KEY constraints
        - FOREIGN KEY ... REFERENCES ... constraints

    Args:
        table_schema: A single entry from Spider's tables.json.

    Returns:
        Multi-line string of CREATE TABLE statements for this database.
    """
    table_names: list[str] = table_schema["table_names_original"]
    column_entries: list[list] = table_schema["column_names_original"]
    column_types: list[str] = table_schema["column_types"]
    primary_keys: list[int] = table_schema.get("primary_keys", [])
    foreign_keys: list[list[int]] = table_schema.get("foreign_keys", [])

    # Build a set of primary key column indices for fast lookup
    pk_set: set[int] = set(primary_keys)

    # Build a mapping: column_index -> (referenced_table, referenced_column)
    # foreign_keys entries are [child_col_idx, parent_col_idx]
    fk_map: dict[int, tuple[str, str]] = {}
    for child_idx, parent_idx in foreign_keys:
        parent_table_idx = column_entries[parent_idx][0]
        parent_col_name = column_entries[parent_idx][1]
        parent_table_name = table_names[parent_table_idx]
        fk_map[child_idx] = (parent_table_name, parent_col_name)

    # Group columns by table
    # columns_by_table[table_idx] = [(col_global_idx, col_name, col_type), ...]
    columns_by_table: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for col_idx, (table_idx, col_name) in enumerate(column_entries):
        if table_idx == -1:
            # Spider's special "*" column — skip it, not part of any table
            continue
        col_type = SPIDER_TYPE_MAP.get(column_types[col_idx], "TEXT")
        columns_by_table[table_idx].append((col_idx, col_name, col_type))

    # Build CREATE TABLE statements
    statements: list[str] = []
    for table_idx, table_name in enumerate(table_names):
        columns = columns_by_table.get(table_idx, [])
        if not columns:
            continue

        lines: list[str] = []
        for col_idx, col_name, col_type in columns:
            col_def = f"  {col_name} {col_type}"
            if col_idx in pk_set:
                col_def += " PRIMARY KEY"
            lines.append(col_def)

        # Add FOREIGN KEY constraints at the end of the table definition
        for col_idx, col_name, _ in columns:
            if col_idx in fk_map:
                ref_table, ref_col = fk_map[col_idx]
                lines.append(
                    f"  FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_col})"
                )

        columns_sql = ",\n".join(lines)
        statements.append(f"CREATE TABLE {table_name} (\n{columns_sql}\n);")

    return "\n\n".join(statements)


def build_schema_lookup(tables_json_path: Path) -> dict[str, str]:
    """Build a db_id → CREATE TABLE schema string mapping.

    Args:
        tables_json_path: Path to Spider's tables.json file.

    Returns:
        Dict mapping each database ID to its reconstructed CREATE TABLE schema.
    """
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables_data: list[dict[str, Any]] = json.load(f)

    schema_lookup: dict[str, str] = {}
    for table_schema in tables_data:
        db_id = table_schema["db_id"]
        schema_lookup[db_id] = build_create_table_statements(table_schema)

    return schema_lookup


# =============================================================================
# Example Conversion
# =============================================================================

INSTRUCTION_TEXT = (
    "Given the database schema, write a SQL query that answers the question."
)


def convert_example(
    example: dict[str, Any],
    schema_lookup: dict[str, str],
) -> dict[str, str] | None:
    """Convert a single Spider example to instruction format.

    Args:
        example: A single entry from train_spider.json or dev.json.
        schema_lookup: Mapping from db_id to CREATE TABLE schema string.

    Returns:
        Instruction-format dict, or None if the example's db_id has no schema
        (this shouldn't happen with a valid Spider download, but we handle it
        defensively).
    """
    db_id: str = example["db_id"]

    if db_id not in schema_lookup:
        print(f"  [WARNING] db_id '{db_id}' not found in tables.json — skipping.")
        return None

    return {
        "instruction": INSTRUCTION_TEXT,
        "schema": schema_lookup[db_id],
        "question": example["question"],
        "output": example["query"],
        "db_id": db_id,
    }


def convert_examples(
    examples: list[dict[str, Any]],
    schema_lookup: dict[str, str],
    source_name: str,
) -> list[dict[str, str]]:
    """Convert a list of Spider examples to instruction format.

    Args:
        examples: List of raw Spider examples.
        schema_lookup: Mapping from db_id to CREATE TABLE schema string.
        source_name: Human-readable source name for progress logging.

    Returns:
        List of converted instruction-format dicts.
    """
    converted: list[dict[str, str]] = []
    skipped = 0

    for example in examples:
        result = convert_example(example, schema_lookup)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    print(f"  {source_name}: {len(converted)} converted, {skipped} skipped")
    return converted


# =============================================================================
# Train / Val Split
# =============================================================================

def split_by_database(
    examples: list[dict[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split examples into train and validation sets BY DATABASE.

    Why split by database instead of by example?
    If we split randomly by example, the model could see training examples
    from database X and then be validated on other examples from the same
    database X. Since the schema is identical, this inflates validation
    metrics — the model has already memorized that schema's patterns.

    Splitting by database ensures the validation set contains databases
    the model has NEVER seen during training, giving a more honest signal
    of generalization.

    Args:
        examples: All converted training examples (instruction format).
        val_ratio: Fraction of *databases* to hold out for validation.
        seed: Random seed for reproducible splitting.

    Returns:
        (train_examples, val_examples) tuple.
    """
    # Group examples by database
    db_to_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for ex in examples:
        db_to_examples[ex["db_id"]].append(ex)

    # Shuffle database IDs deterministically
    db_ids = sorted(db_to_examples.keys())  # Sort first for determinism
    rng = random.Random(seed)
    rng.shuffle(db_ids)

    # Split databases
    n_val_dbs = max(1, int(len(db_ids) * val_ratio))
    val_db_ids = set(db_ids[:n_val_dbs])
    train_db_ids = set(db_ids[n_val_dbs:])

    train_examples: list[dict[str, str]] = []
    val_examples: list[dict[str, str]] = []

    for db_id in sorted(train_db_ids):
        train_examples.extend(db_to_examples[db_id])
    for db_id in sorted(val_db_ids):
        val_examples.extend(db_to_examples[db_id])

    print(f"  Split: {len(train_db_ids)} train DBs ({len(train_examples)} examples), "
          f"{len(val_db_ids)} val DBs ({len(val_examples)} examples)")

    return train_examples, val_examples


# =============================================================================
# I/O
# =============================================================================

def save_json(data: list[dict[str, str]], output_path: Path) -> None:
    """Save a list of dicts to a JSON file (one JSON array, pretty-printed).

    Args:
        data: List of instruction-format dicts to save.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} examples to {output_path}")


def load_spider_json(path: Path) -> list[dict[str, Any]]:
    """Load a Spider JSON file (train_spider.json or dev.json).

    Args:
        path: Path to the JSON file.

    Returns:
        List of raw Spider example dicts.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Spider file not found: {path}\n"
            f"Make sure you've downloaded the Spider dataset into the correct directory.\n"
            f"Expected structure:\n"
            f"  {path.parent}/\n"
            f"    ├── train_spider.json\n"
            f"    ├── dev.json\n"
            f"    ├── tables.json\n"
            f"    └── database/\n"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Statistics
# =============================================================================

def print_dataset_stats(examples: list[dict[str, str]], name: str) -> None:
    """Print basic statistics about a converted dataset split.

    Useful for sanity-checking after conversion — helps catch issues like
    missing schemas or skewed database distributions.

    Args:
        examples: List of instruction-format examples.
        name: Human-readable name for this split (e.g., "train", "val").
    """
    if not examples:
        print(f"\n  [{name}] Empty — no examples.")
        return

    db_counts: dict[str, int] = defaultdict(int)
    schema_lengths: list[int] = []
    query_lengths: list[int] = []

    for ex in examples:
        db_counts[ex["db_id"]] += 1
        schema_lengths.append(len(ex["schema"]))
        query_lengths.append(len(ex["output"]))

    print(f"\n  [{name}] Statistics:")
    print(f"    Total examples:     {len(examples)}")
    print(f"    Unique databases:   {len(db_counts)}")
    print(f"    Schema length:      min={min(schema_lengths)}, "
          f"max={max(schema_lengths)}, "
          f"avg={sum(schema_lengths) / len(schema_lengths):.0f} chars")
    print(f"    Query length:       min={min(query_lengths)}, "
          f"max={max(query_lengths)}, "
          f"avg={sum(query_lengths) / len(query_lengths):.0f} chars")
    print(f"    Top 5 databases:    {sorted(db_counts.items(), key=lambda x: -x[1])[:5]}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Spider dataset to instruction-tuning format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/convert_spider_to_instruction_format.py \\\n"
            "      --spider_dir data/spider \\\n"
            "      --output_dir data/processed \\\n"
            "      --val_ratio 0.1 \\\n"
            "      --seed 42\n"
        ),
    )
    parser.add_argument(
        "--spider_dir",
        type=str,
        required=True,
        help="Path to the raw Spider dataset directory (contains train_spider.json, "
             "dev.json, tables.json, database/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed JSON files.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of databases to hold out for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for Spider-to-instruction-format conversion."""
    args = parse_args()
    spider_dir = Path(args.spider_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Spider → Instruction Format Conversion")
    print("=" * 70)

    # --- Step 1: Load schemas ---
    print("\n[1/4] Loading database schemas from tables.json...")
    tables_path = spider_dir / "tables.json"
    schema_lookup = build_schema_lookup(tables_path)
    print(f"  Loaded schemas for {len(schema_lookup)} databases.")

    # --- Step 2: Convert training examples ---
    print("\n[2/4] Converting training examples...")
    train_raw = load_spider_json(spider_dir / "train_spider.json")
    train_converted = convert_examples(train_raw, schema_lookup, "train_spider.json")

    # --- Step 3: Split train into train/val (by database) ---
    print("\n[3/4] Splitting train set by database...")
    train_examples, val_examples = split_by_database(
        train_converted,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # --- Step 4: Convert dev set (our held-out test set) ---
    print("\n[4/4] Converting dev set (held-out test set)...")
    dev_raw = load_spider_json(spider_dir / "dev.json")
    dev_examples = convert_examples(dev_raw, schema_lookup, "dev.json")

    # --- Print statistics ---
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    print_dataset_stats(train_examples, "train")
    print_dataset_stats(val_examples, "val")
    print_dataset_stats(dev_examples, "dev (test)")

    # --- Save outputs ---
    print("\n" + "=" * 70)
    print("Saving processed files")
    print("=" * 70)
    save_json(train_examples, output_dir / "train.json")
    save_json(val_examples, output_dir / "val.json")
    save_json(dev_examples, output_dir / "dev.json")

    # --- Save metadata for reproducibility ---
    metadata = {
        "source": str(spider_dir),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "counts": {
            "train": len(train_examples),
            "val": len(val_examples),
            "dev_test": len(dev_examples),
        },
        "num_databases": {
            "total": len(schema_lookup),
        },
    }
    save_json([metadata], output_dir / "conversion_metadata.json")

    print("\n✓ Conversion complete.")
    print(f"  Train: {output_dir / 'train.json'}")
    print(f"  Val:   {output_dir / 'val.json'}")
    print(f"  Test:  {output_dir / 'dev.json'}")
    print(f"  Meta:  {output_dir / 'conversion_metadata.json'}")


if __name__ == "__main__":
    main()
