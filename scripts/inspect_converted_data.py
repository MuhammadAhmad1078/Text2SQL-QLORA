#!/usr/bin/env python3
"""
Inspect converted Spider examples for manual quality review.

This script loads processed JSON files produced by convert_spider_to_instruction_format.py
and displays them in a human-readable format for manual sanity checking.

Purpose:
    - Verify the conversion looks correct before using the data for training
    - Spot schema-linking issues (e.g., missing tables, wrong column types)
    - Identify ambiguous questions or edge-case queries
    - Take notes on patterns that might cause problems during fine-tuning

Usage:
    # Show 20 random examples from the training set:
    python scripts/inspect_converted_data.py \
        --data_path data/processed/train.json \
        --num_examples 20

    # Show examples from a specific database:
    python scripts/inspect_converted_data.py \
        --data_path data/processed/train.json \
        --filter_db concert_singer

    # Show all examples (pipe to less for paging):
    python scripts/inspect_converted_data.py \
        --data_path data/processed/train.json \
        --num_examples 0

    # Show overall statistics only (no individual examples):
    python scripts/inspect_converted_data.py \
        --data_path data/processed/train.json \
        --stats_only
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
# Display Helpers
# =============================================================================

def display_example(example: dict[str, str], index: int) -> None:
    """Pretty-print a single converted example for manual inspection.

    Output is designed to be scannable: clear section headers, bordered
    schema and SQL blocks, and enough whitespace to distinguish examples.

    Args:
        example: A single instruction-format example dict.
        index: 1-based index for display numbering.
    """
    separator = "─" * 70
    print(f"\n{'═' * 70}")
    print(f"  Example {index}  │  db_id: {example['db_id']}")
    print(f"{'═' * 70}")

    print(f"\n  ▸ INSTRUCTION:")
    print(f"    {example['instruction']}")

    print(f"\n  ▸ SCHEMA:")
    # Indent each line of the schema for readability
    for line in example["schema"].split("\n"):
        print(f"    {line}")

    print(f"\n  ▸ QUESTION:")
    print(f"    {example['question']}")

    print(f"\n  ▸ GOLD SQL:")
    print(f"    {example['output']}")

    # Quick flags for potential issues
    flags: list[str] = []
    query_upper = example["output"].upper()

    if "JOIN" in query_upper:
        flags.append("JOINS")
    if "SUBQUERY" in query_upper or "SELECT" in query_upper.split("FROM", 1)[-1] if "FROM" in query_upper else False:
        # Rough heuristic: if SELECT appears after the first FROM, likely a subquery
        after_from = query_upper.split("FROM", 1)[-1] if "FROM" in query_upper else ""
        if "SELECT" in after_from:
            flags.append("SUBQUERY")
    if any(agg in query_upper for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
        flags.append("AGGREGATION")
    if "GROUP BY" in query_upper:
        flags.append("GROUP BY")
    if "HAVING" in query_upper:
        flags.append("HAVING")
    if "ORDER BY" in query_upper:
        flags.append("ORDER BY")
    if "INTERSECT" in query_upper or "UNION" in query_upper or "EXCEPT" in query_upper:
        flags.append("SET OPS")
    if "LIKE" in query_upper:
        flags.append("LIKE")
    if query_upper.count("SELECT") > 1:
        flags.append("NESTED")

    if flags:
        print(f"\n  ▸ SQL FEATURES: {', '.join(flags)}")

    print(f"\n{separator}")


def display_statistics(examples: list[dict[str, str]], data_path: str) -> None:
    """Display aggregate statistics about the dataset.

    Args:
        examples: All loaded examples.
        data_path: Path string for display purposes.
    """
    print(f"\n{'═' * 70}")
    print(f"  Dataset Statistics: {data_path}")
    print(f"{'═' * 70}")

    if not examples:
        print("  (empty dataset)")
        return

    # Database distribution
    db_counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        db_counts[ex["db_id"]] += 1

    # Schema and query length distributions
    schema_lengths = [len(ex["schema"]) for ex in examples]
    query_lengths = [len(ex["output"]) for ex in examples]
    question_lengths = [len(ex["question"].split()) for ex in examples]

    # SQL feature distribution
    feature_counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        q = ex["output"].upper()
        if "JOIN" in q:
            feature_counts["JOIN"] += 1
        if any(a in q for a in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
            feature_counts["Aggregation"] += 1
        if "GROUP BY" in q:
            feature_counts["GROUP BY"] += 1
        if "HAVING" in q:
            feature_counts["HAVING"] += 1
        if "ORDER BY" in q:
            feature_counts["ORDER BY"] += 1
        if q.count("SELECT") > 1:
            feature_counts["Nested/Subquery"] += 1
        if "INTERSECT" in q or "UNION" in q or "EXCEPT" in q:
            feature_counts["Set Operations"] += 1
        if "LIKE" in q:
            feature_counts["LIKE"] += 1

    print(f"\n  Total examples:       {len(examples)}")
    print(f"  Unique databases:     {len(db_counts)}")

    print(f"\n  Schema length (chars):")
    print(f"    Min: {min(schema_lengths):>6}  "
          f"Max: {max(schema_lengths):>6}  "
          f"Avg: {sum(schema_lengths) / len(schema_lengths):>6.0f}")

    print(f"\n  Query length (chars):")
    print(f"    Min: {min(query_lengths):>6}  "
          f"Max: {max(query_lengths):>6}  "
          f"Avg: {sum(query_lengths) / len(query_lengths):>6.0f}")

    print(f"\n  Question length (words):")
    print(f"    Min: {min(question_lengths):>6}  "
          f"Max: {max(question_lengths):>6}  "
          f"Avg: {sum(question_lengths) / len(question_lengths):>6.0f}")

    print(f"\n  SQL Feature Distribution:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        pct = count / len(examples) * 100
        bar = "█" * int(pct / 2)
        print(f"    {feature:<20} {count:>5} ({pct:>5.1f}%) {bar}")

    print(f"\n  Database Distribution (top 10 / {len(db_counts)}):")
    for db_id, count in sorted(db_counts.items(), key=lambda x: -x[1])[:10]:
        pct = count / len(examples) * 100
        print(f"    {db_id:<30} {count:>4} ({pct:>5.1f}%)")

    # Token length estimation (rough: 1 token ≈ 4 chars)
    # This helps estimate if examples fit within max_seq_length
    full_lengths = [
        len(ex["instruction"]) + len(ex["schema"]) + len(ex["question"]) + len(ex["output"])
        for ex in examples
    ]
    est_tokens = [l // 4 for l in full_lengths]
    over_1024 = sum(1 for t in est_tokens if t > 1024)
    over_512 = sum(1 for t in est_tokens if t > 512)

    print(f"\n  Estimated Token Lengths (1 token ≈ 4 chars):")
    print(f"    Min: {min(est_tokens):>6}  "
          f"Max: {max(est_tokens):>6}  "
          f"Avg: {sum(est_tokens) / len(est_tokens):>6.0f}")
    print(f"    > 512 tokens:  {over_512:>5} ({over_512 / len(examples) * 100:.1f}%)")
    print(f"    > 1024 tokens: {over_1024:>5} ({over_1024 / len(examples) * 100:.1f}%)")
    if over_1024 > 0:
        print(f"    ⚠ {over_1024} examples may be truncated with max_seq_length=1024")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect converted Spider examples for quality review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to a processed JSON file (e.g., data/processed/train.json).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of random examples to display (0 = all). Default: 20.",
    )
    parser.add_argument(
        "--filter_db",
        type=str,
        default=None,
        help="If set, only show examples from this database (e.g., 'concert_singer').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling examples. Default: 42.",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only display statistics, don't show individual examples.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for data inspection."""
    args = parse_args()
    data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        sys.exit(1)

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        examples: list[dict[str, str]] = json.load(f)

    print(f"Loaded {len(examples)} examples from {data_path}")

    # Apply database filter if specified
    if args.filter_db:
        examples = [ex for ex in examples if ex["db_id"] == args.filter_db]
        print(f"Filtered to {len(examples)} examples from db_id='{args.filter_db}'")
        if not examples:
            print(f"No examples found for db_id='{args.filter_db}'.")
            # Show available db_ids
            with open(data_path, "r", encoding="utf-8") as f:
                all_examples = json.load(f)
            db_ids = sorted(set(ex["db_id"] for ex in all_examples))
            print(f"Available db_ids ({len(db_ids)}):")
            for db_id in db_ids[:20]:
                print(f"  - {db_id}")
            if len(db_ids) > 20:
                print(f"  ... and {len(db_ids) - 20} more")
            sys.exit(0)

    # Always show statistics
    display_statistics(examples, str(data_path))

    # Show individual examples unless --stats_only
    if not args.stats_only:
        # Sample or show all
        if args.num_examples > 0 and args.num_examples < len(examples):
            rng = random.Random(args.seed)
            sampled = rng.sample(examples, args.num_examples)
            print(f"\nShowing {args.num_examples} randomly sampled examples "
                  f"(seed={args.seed}):")
        else:
            sampled = examples
            print(f"\nShowing all {len(sampled)} examples:")

        for i, example in enumerate(sampled, 1):
            display_example(example, i)

    print(f"\n{'═' * 70}")
    print("  Inspection complete.")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
