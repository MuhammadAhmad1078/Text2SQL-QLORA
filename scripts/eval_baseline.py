#!/usr/bin/env python3
"""
Baseline evaluation: zero-shot and few-shot prompting against the base model.

This script evaluates the base model (no fine-tuning) on the Spider dev set
using two prompting strategies:
    1. Zero-shot: just the schema + question, no examples
    2. Few-shot: 3-5 in-context examples prepended to the prompt

Both modes compute:
    - Exact-match accuracy (normalized string comparison of predicted vs gold SQL)
    - Execution accuracy (run both queries against the actual SQLite database,
      compare result sets)

This establishes the baseline numbers that fine-tuning must beat. Without these,
any fine-tuned results are meaningless — you can't claim improvement without
a comparison point.

Design decisions:
    1. Model is loaded in 4-bit (NF4) quantization to fit on a T4 (16GB VRAM).
       This matches the quantization used during QLoRA training, so the baseline
       is a fair comparison — same model weights, same precision.
    2. Few-shot examples are sampled from the TRAINING set (never from test).
       We use a fixed seed so the same examples are selected every run.
    3. Execution accuracy uses a timeout per query to prevent infinite loops
       from malformed SQL. Result sets are compared as sets of tuples (order-
       independent) to avoid false negatives from different ORDER BY behavior.
    4. The model's chat template is used via tokenizer.apply_chat_template()
       to ensure the prompt format matches what the model was instruction-tuned
       with. Skipping this would degrade performance.
    5. Results are saved as JSON in a standardized format so they can be directly
       compared with fine-tuned eval results later.

Usage:
    # Zero-shot baseline
    python scripts/eval_baseline.py \
        --model microsoft/Phi-3-mini-4k-instruct \
        --mode zero-shot \
        --test_set data/processed/dev.json \
        --db_dir data/spider/database \
        --output_dir outputs/baseline

    # Few-shot baseline (5 examples)
    python scripts/eval_baseline.py \
        --model microsoft/Phi-3-mini-4k-instruct \
        --mode few-shot \
        --num_few_shot 5 \
        --train_set data/processed/train.json \
        --test_set data/processed/dev.json \
        --db_dir data/spider/database \
        --output_dir outputs/baseline

    # Both modes in sequence
    python scripts/eval_baseline.py \
        --model microsoft/Phi-3-mini-4k-instruct \
        --mode both \
        --test_set data/processed/dev.json \
        --db_dir data/spider/database \
        --output_dir outputs/baseline
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# =============================================================================
# SQL Execution & Comparison (for execution accuracy)
# =============================================================================

EXEC_TIMEOUT_SECONDS = 30  # Max time per query execution


def execute_sql(
    db_path: str,
    sql: str,
    timeout: int = EXEC_TIMEOUT_SECONDS,
) -> tuple[list[tuple] | None, str | None]:
    """Execute a SQL query against a SQLite database and return results.

    Args:
        db_path: Path to the SQLite database file.
        sql: SQL query to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        (results, error) tuple. On success, results is a list of tuples and
        error is None. On failure, results is None and error is the error message.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        # Prevent excessively slow queries
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def compare_result_sets(
    gold_results: list[tuple] | None,
    pred_results: list[tuple] | None,
) -> bool:
    """Compare two SQL query result sets for equality.

    Comparison is order-independent (set comparison) because two logically
    equivalent queries may return rows in different orders if no ORDER BY
    is specified.

    For result sets with duplicate rows, we use multiset comparison
    (sorted lists) to handle cases like COUNT queries returning the same
    value multiple times.

    Args:
        gold_results: Result set from the gold SQL query.
        pred_results: Result set from the predicted SQL query.

    Returns:
        True if result sets are equivalent, False otherwise.
    """
    if gold_results is None or pred_results is None:
        return False

    # Normalize: convert to sorted lists of tuples for multiset comparison
    # This handles duplicate rows correctly while being order-independent
    try:
        gold_sorted = sorted([tuple(str(v) for v in row) for row in gold_results])
        pred_sorted = sorted([tuple(str(v) for v in row) for row in pred_results])
        return gold_sorted == pred_sorted
    except TypeError:
        # Fallback for unhashable/uncomparable types
        return gold_results == pred_results


def compute_execution_accuracy(
    db_path: str,
    gold_sql: str,
    pred_sql: str,
) -> dict[str, Any]:
    """Compute execution accuracy for a single example.

    Runs both gold and predicted SQL against the database and compares
    result sets.

    Args:
        db_path: Path to the SQLite database file.
        gold_sql: The ground-truth SQL query.
        pred_sql: The model-predicted SQL query.

    Returns:
        Dict with keys:
            - 'exec_match': bool — whether result sets match
            - 'gold_error': error string or None
            - 'pred_error': error string or None
            - 'gold_rows': number of result rows (or -1 on error)
            - 'pred_rows': number of result rows (or -1 on error)
    """
    gold_results, gold_error = execute_sql(db_path, gold_sql)
    pred_results, pred_error = execute_sql(db_path, pred_sql)

    return {
        "exec_match": compare_result_sets(gold_results, pred_results),
        "gold_error": gold_error,
        "pred_error": pred_error,
        "gold_rows": len(gold_results) if gold_results is not None else -1,
        "pred_rows": len(pred_results) if pred_results is not None else -1,
    }


# =============================================================================
# SQL Normalization (for exact-match accuracy)
# =============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact-match comparison.

    Applies aggressive normalization to reduce trivial formatting differences:
    - Uppercase all keywords
    - Collapse whitespace
    - Strip trailing semicolons and whitespace
    - Lowercase identifiers

    Note: This is intentionally NOT semantic equivalence — textually different
    but logically equivalent queries will still fail exact match. That's by
    design; execution accuracy captures semantic equivalence.

    Args:
        sql: Raw SQL string.

    Returns:
        Normalized SQL string.
    """
    try:
        import sqlparse
        formatted = sqlparse.format(
            sql,
            keyword_case="upper",
            identifier_case="lower",
            strip_comments=True,
            reindent=False,
        )
    except ImportError:
        # Fallback if sqlparse not available
        formatted = sql

    # Remove trailing semicolons
    formatted = formatted.rstrip(";").strip()
    # Collapse whitespace
    formatted = re.sub(r"\s+", " ", formatted)
    return formatted.strip()


def compute_exact_match(gold_sql: str, pred_sql: str) -> bool:
    """Check if predicted SQL matches gold SQL after normalization.

    Args:
        gold_sql: Ground-truth SQL query.
        pred_sql: Model-predicted SQL query.

    Returns:
        True if normalized queries match exactly.
    """
    return normalize_sql(gold_sql) == normalize_sql(pred_sql)


# =============================================================================
# SQL Extraction from Model Output
# =============================================================================

def extract_sql_from_response(response: str) -> str:
    """Extract the SQL query from the model's generated response.

    The model may generate additional text around the SQL (explanations,
    markdown formatting, multiple statements). This function extracts
    just the SQL query.

    Extraction strategy (in order of priority):
    1. If response contains a SQL code block (```sql ... ```), extract it
    2. If response starts with SELECT/INSERT/UPDATE/DELETE/WITH/CREATE, take
       everything up to the first semicolon or blank line
    3. Otherwise, take the first line that looks like SQL

    Args:
        response: The full model response text.

    Returns:
        Extracted SQL query string (may be empty if no SQL found).
    """
    response = response.strip()

    # Strategy 1: Extract from markdown code blocks
    code_block_match = re.search(
        r"```(?:sql)?\s*\n?(.*?)```",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    # Strategy 2: Response starts with a SQL keyword
    sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE")
    upper_response = response.upper().lstrip()
    if any(upper_response.startswith(kw) for kw in sql_keywords):
        # Take up to first semicolon, double newline, or end
        match = re.match(r"(.*?)(?:;|\n\n|$)", response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Strategy 3: Find the first line starting with a SQL keyword
    for line in response.split("\n"):
        stripped = line.strip()
        if any(stripped.upper().startswith(kw) for kw in sql_keywords):
            return stripped.rstrip(";").strip()

    # Fallback: return the whole response (will likely fail exact match but
    # let execution accuracy judge it)
    return response.strip()


# =============================================================================
# Prompt Formatting
# =============================================================================

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate the correct SQL query. Output ONLY the SQL query, nothing else."
)


def format_zero_shot_prompt(schema: str, question: str) -> str:
    """Format a zero-shot prompt for the model.

    Args:
        schema: CREATE TABLE statements for the database.
        question: Natural language question.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Given the following database schema, write a SQL query that "
        f"answers the question.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"SQL:"
    )


def format_few_shot_prompt(
    schema: str,
    question: str,
    examples: list[dict[str, str]],
) -> str:
    """Format a few-shot prompt with in-context examples.

    Args:
        schema: CREATE TABLE statements for the target question's database.
        question: Natural language question to answer.
        examples: List of dicts with 'schema', 'question', 'output' keys.

    Returns:
        Formatted few-shot prompt string.
    """
    parts = [
        "Given a database schema and a question, write the SQL query that "
        "answers the question. Here are some examples:\n"
    ]

    for i, ex in enumerate(examples, 1):
        parts.append(f"\n### Example {i}")
        parts.append(f"Schema:\n{ex['schema']}\n")
        parts.append(f"Question: {ex['question']}\n")
        parts.append(f"SQL: {ex['output']}\n")

    parts.append("\n### Your Turn")
    parts.append(f"Schema:\n{schema}\n")
    parts.append(f"Question: {question}\n")
    parts.append("SQL:")

    return "\n".join(parts)


def select_few_shot_examples(
    train_data: list[dict[str, str]],
    num_examples: int,
    seed: int,
    exclude_db_id: str | None = None,
) -> list[dict[str, str]]:
    """Select few-shot in-context examples from the training set.

    Selection strategy: random sample from training set, excluding examples
    from the same database as the test example (to prevent schema leakage
    in the few-shot context).

    We also prefer shorter schemas to keep the prompt within token limits.
    Long schemas eat into the context window and can cause truncation.

    Args:
        train_data: All training examples.
        num_examples: Number of few-shot examples to select.
        seed: Random seed for reproducibility.
        exclude_db_id: If set, exclude examples from this database.

    Returns:
        List of selected example dicts.
    """
    # Filter out examples from the same database
    candidates = train_data
    if exclude_db_id:
        candidates = [ex for ex in candidates if ex["db_id"] != exclude_db_id]

    # Sort by schema length (prefer shorter schemas to save tokens)
    # then take a random sample from the shortest ~50%
    candidates_sorted = sorted(candidates, key=lambda x: len(x["schema"]))
    pool = candidates_sorted[: len(candidates_sorted) // 2]

    if len(pool) < num_examples:
        pool = candidates  # Fallback to all candidates

    rng = random.Random(seed)
    return rng.sample(pool, min(num_examples, len(pool)))


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
) -> tuple[Any, Any]:
    """Load the base model in 4-bit quantization with its tokenizer.

    We load in 4-bit NF4 (same quantization as QLoRA training) so the
    baseline is a fair comparison — identical model weights and precision.

    Args:
        model_name: HuggingFace model identifier
            (e.g., 'microsoft/Phi-3-mini-4k-instruct').
        device_map: Device mapping strategy. 'auto' places layers
            across available GPUs/CPU automatically.

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,  # Required for Phi-3
        padding_side="left",     # Left-pad for batch generation
    )

    # Set pad token if not already set (Phi-3 may not have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model in 4-bit NF4 quantization: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # float16 for T4 compatibility
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,  # Required for Phi-3
        attn_implementation="eager",  # Avoids flash-attention issues on T4
    )

    model.eval()  # Set to eval mode (disables dropout)

    print(f"Model loaded. Device map: {model.hf_device_map}")
    return model, tokenizer


# =============================================================================
# Generation
# =============================================================================

def generate_sql(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Generate a SQL query from a prompt using the model.

    Uses the model's chat template to format the input correctly for
    instruction-tuned models. Temperature=0 (greedy decoding) ensures
    deterministic, reproducible outputs.

    Args:
        model: The loaded language model.
        tokenizer: The model's tokenizer.
        prompt: The formatted prompt (schema + question).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature. 0 = greedy (deterministic).

    Returns:
        The generated SQL query string.
    """
    # Format using the model's chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # apply_chat_template handles the model-specific formatting
    # (ChatML for Phi-3, Llama chat format for Llama, etc.)
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=3072,  # Leave room for generation within 4k context
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # Greedy decoding for reproducibility
            temperature=None,         # Not used with do_sample=False
            top_p=None,               # Not used with do_sample=False
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return extract_sql_from_response(response)


# =============================================================================
# Evaluation Loop
# =============================================================================

def evaluate_model(
    model: Any,
    tokenizer: Any,
    test_data: list[dict[str, str]],
    db_dir: str,
    mode: str,
    train_data: list[dict[str, str]] | None = None,
    num_few_shot: int = 5,
    seed: int = 42,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Run evaluation on the test set and compute metrics.

    Args:
        model: The loaded language model.
        tokenizer: The model's tokenizer.
        test_data: List of test examples (instruction format).
        db_dir: Path to Spider's database/ directory.
        mode: 'zero-shot' or 'few-shot'.
        train_data: Training data (required for few-shot mode).
        num_few_shot: Number of few-shot examples (only for few-shot mode).
        seed: Random seed for few-shot example selection.
        max_examples: If set, evaluate only this many examples (for debugging).

    Returns:
        Results dict with per-example predictions, aggregate metrics,
        and metadata.
    """
    if mode == "few-shot" and train_data is None:
        raise ValueError("--train_set is required for few-shot mode")

    # Subset for debugging if requested
    examples = test_data[:max_examples] if max_examples else test_data

    results: list[dict[str, Any]] = []
    exact_matches = 0
    exec_matches = 0
    exec_errors = 0
    total = len(examples)

    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {mode} mode ({total} examples)")
    print(f"{'=' * 70}\n")

    for i, example in enumerate(tqdm(examples, desc=f"{mode} eval")):
        db_id = example["db_id"]
        schema = example["schema"]
        question = example["question"]
        gold_sql = example["output"]

        # Format prompt based on mode
        if mode == "zero-shot":
            prompt = format_zero_shot_prompt(schema, question)
        else:
            few_shot_examples = select_few_shot_examples(
                train_data,
                num_examples=num_few_shot,
                seed=seed + i,  # Different examples for each test query
                exclude_db_id=db_id,
            )
            prompt = format_few_shot_prompt(schema, question, few_shot_examples)

        # Generate prediction
        start_time = time.time()
        pred_sql = generate_sql(model, tokenizer, prompt)
        gen_time = time.time() - start_time

        # Compute exact match
        em = compute_exact_match(gold_sql, pred_sql)
        if em:
            exact_matches += 1

        # Compute execution accuracy
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_path):
            exec_result = compute_execution_accuracy(db_path, gold_sql, pred_sql)
            if exec_result["exec_match"]:
                exec_matches += 1
            if exec_result["pred_error"]:
                exec_errors += 1
        else:
            exec_result = {
                "exec_match": False,
                "gold_error": f"Database not found: {db_path}",
                "pred_error": f"Database not found: {db_path}",
                "gold_rows": -1,
                "pred_rows": -1,
            }

        # Store per-example result
        result = {
            "index": i,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": em,
            "exec_match": exec_result["exec_match"],
            "pred_error": exec_result["pred_error"],
            "generation_time_s": round(gen_time, 2),
        }
        results.append(result)

        # Progress logging every 50 examples
        if (i + 1) % 50 == 0 or i == 0:
            em_pct = exact_matches / (i + 1) * 100
            ex_pct = exec_matches / (i + 1) * 100
            print(f"\n  [{i+1}/{total}] Running EM: {em_pct:.1f}%, "
                  f"Exec: {ex_pct:.1f}%, "
                  f"Errors: {exec_errors}")

    # Compute aggregate metrics
    metrics = {
        "mode": mode,
        "total_examples": total,
        "exact_match_accuracy": round(exact_matches / total * 100, 2),
        "execution_accuracy": round(exec_matches / total * 100, 2),
        "exact_match_count": exact_matches,
        "execution_match_count": exec_matches,
        "execution_error_count": exec_errors,
        "avg_generation_time_s": round(
            sum(r["generation_time_s"] for r in results) / total, 2
        ),
    }

    return {
        "metrics": metrics,
        "predictions": results,
        "config": {
            "mode": mode,
            "num_few_shot": num_few_shot if mode == "few-shot" else 0,
            "seed": seed,
            "total_examples": total,
        },
    }


# =============================================================================
# Results Display & Saving
# =============================================================================

def print_results(metrics: dict[str, Any]) -> None:
    """Pretty-print evaluation metrics.

    Args:
        metrics: The metrics dict from evaluate_model().
    """
    print(f"\n{'═' * 70}")
    print(f"  RESULTS: {metrics['mode']}")
    print(f"{'═' * 70}")
    print(f"  Total examples:        {metrics['total_examples']}")
    print(f"  Exact Match Accuracy:  {metrics['exact_match_accuracy']:.2f}%"
          f"  ({metrics['exact_match_count']}/{metrics['total_examples']})")
    print(f"  Execution Accuracy:    {metrics['execution_accuracy']:.2f}%"
          f"  ({metrics['execution_match_count']}/{metrics['total_examples']})")
    print(f"  Execution Errors:      {metrics['execution_error_count']}"
          f"  ({metrics['execution_error_count']/metrics['total_examples']*100:.1f}%)")
    print(f"  Avg Generation Time:   {metrics['avg_generation_time_s']:.2f}s")
    print(f"{'═' * 70}\n")


def save_results(
    results: dict[str, Any],
    output_dir: str,
    mode: str,
) -> None:
    """Save evaluation results to JSON files.

    Saves two files:
    1. Full results (predictions + metrics) — for detailed analysis
    2. Metrics-only summary — for quick comparison in the README table

    Args:
        results: Full results dict from evaluate_model().
        output_dir: Directory to save output files.
        mode: 'zero-shot' or 'few-shot' (used in filename).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Full results
    full_path = output_path / f"baseline_{mode}_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Full results saved to: {full_path}")

    # Metrics summary
    summary_path = output_path / f"baseline_{mode}_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, indent=2)
    print(f"  Metrics summary saved to: {summary_path}")


def print_error_examples(
    predictions: list[dict[str, Any]],
    num_examples: int = 10,
) -> None:
    """Print a sample of incorrect predictions for manual inspection.

    Shows examples where execution accuracy failed, which are the most
    interesting for error analysis.

    Args:
        predictions: List of per-example prediction dicts.
        num_examples: Number of error examples to display.
    """
    errors = [p for p in predictions if not p["exec_match"]]
    sample = errors[:num_examples]

    print(f"\n{'═' * 70}")
    print(f"  Sample of {len(sample)} incorrect predictions (of {len(errors)} total)")
    print(f"{'═' * 70}")

    for p in sample:
        print(f"\n  [{p['index']}] db_id={p['db_id']}")
        print(f"  Question: {p['question']}")
        print(f"  Gold SQL: {p['gold_sql']}")
        print(f"  Pred SQL: {p['pred_sql']}")
        if p.get("pred_error"):
            print(f"  Error:    {p['pred_error']}")
        print(f"  EM: {p['exact_match']} | Exec: {p['exec_match']}")
        print(f"  {'─' * 60}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline evaluation: zero-shot and few-shot prompting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HuggingFace model name (default: microsoft/Phi-3-mini-4k-instruct).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero-shot", "few-shot", "both"],
        default="both",
        help="Evaluation mode. 'both' runs zero-shot then few-shot.",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        required=True,
        help="Path to test set JSON (data/processed/dev.json).",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        default=None,
        help="Path to training set JSON (for few-shot examples). "
             "Required if --mode is 'few-shot' or 'both'.",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        required=True,
        help="Path to Spider's database/ directory (for execution accuracy).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baseline",
        help="Directory to save results (default: outputs/baseline).",
    )
    parser.add_argument(
        "--num_few_shot",
        type=int,
        default=5,
        help="Number of in-context examples for few-shot mode (default: 5).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="If set, only evaluate this many examples (for debugging).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for baseline evaluation."""
    args = parse_args()

    # Validate args
    if args.mode in ("few-shot", "both") and args.train_set is None:
        print("Error: --train_set is required for few-shot or both modes.")
        sys.exit(1)

    # Load test data
    print(f"Loading test data from: {args.test_set}")
    with open(args.test_set, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  Loaded {len(test_data)} test examples.")

    # Load training data (for few-shot)
    train_data = None
    if args.train_set:
        print(f"Loading training data from: {args.train_set}")
        with open(args.train_set, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        print(f"  Loaded {len(train_data)} training examples.")

    # Verify database directory
    if not os.path.isdir(args.db_dir):
        print(f"Error: Database directory not found: {args.db_dir}")
        print("  Make sure the Spider dataset with SQLite databases is available.")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Determine which modes to run
    modes = ["zero-shot", "few-shot"] if args.mode == "both" else [args.mode]

    all_metrics = {}

    for mode in modes:
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            db_dir=args.db_dir,
            mode=mode,
            train_data=train_data,
            num_few_shot=args.num_few_shot,
            seed=args.seed,
            max_examples=args.max_examples,
        )

        print_results(results["metrics"])
        print_error_examples(results["predictions"])
        save_results(results, args.output_dir, mode)
        all_metrics[mode] = results["metrics"]

    # Print comparison table if both modes were run
    if len(all_metrics) == 2:
        print(f"\n{'═' * 70}")
        print("  COMPARISON TABLE")
        print(f"{'═' * 70}")
        print(f"  {'Mode':<20} {'Exact Match':>15} {'Execution Acc':>15}")
        print(f"  {'─' * 50}")
        for mode, metrics in all_metrics.items():
            print(f"  {mode:<20} {metrics['exact_match_accuracy']:>14.2f}% "
                  f"{metrics['execution_accuracy']:>14.2f}%")
        print(f"{'═' * 70}\n")

    print("✓ Baseline evaluation complete.")


if __name__ == "__main__":
    main()
