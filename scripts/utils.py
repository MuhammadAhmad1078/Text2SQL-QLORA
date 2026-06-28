"""
Shared utilities for the Text-to-SQL QLoRA project.

This module contains helper functions used across multiple scripts:
- Config loading and merging (YAML base + sweep overrides + CLI args)
- Prompt formatting (consistent instruction template across train/eval/serve)
- SQL normalization (for exact-match comparison)
- Common type definitions

Design decision: Centralizing these here avoids duplication between
eval_baseline.py, eval_finetuned.py, train_qlora.py, and the serving app.
Any change to the prompt template or SQL normalization logic propagates
automatically to all scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
import sqlparse


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a nested dict.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two config dicts. Values in `override` take precedence.

    This allows sweep configs to only specify the keys they change,
    inheriting everything else from the base config.

    Args:
        base: The base/default config dict.
        override: The override config dict (e.g., a sweep config).

    Returns:
        A new dict with base values overridden where specified.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_with_overrides(
    default_path: str | Path,
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the default config, optionally merging a sweep override file.

    Usage pattern in training script:
        config = load_config_with_overrides("configs/default.yaml", args.config)

    Args:
        default_path: Path to the base config (configs/default.yaml).
        override_path: Optional path to a sweep config that overrides specific keys.

    Returns:
        The merged config dict.
    """
    config = load_config(default_path)
    if override_path is not None:
        overrides = load_config(override_path)
        config = merge_configs(config, overrides)
    return config


# =============================================================================
# Prompt Formatting
# =============================================================================

DEFAULT_PROMPT_TEMPLATE = (
    "Given the following database schema, write a SQL query that answers the question.\n"
    "\n"
    "Schema:\n"
    "{schema}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "SQL:"
)

FEW_SHOT_EXAMPLE_TEMPLATE = (
    "Schema:\n"
    "{schema}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "SQL: {sql}\n"
)


def format_prompt(
    schema: str,
    question: str,
    template: str | None = None,
) -> str:
    """Format a single text-to-SQL prompt using the instruction template.

    This is the single source of truth for prompt formatting — used in
    data conversion, baseline eval, fine-tuned eval, and serving.

    Args:
        schema: The database schema (CREATE TABLE statements).
        question: The natural language question.
        template: Optional custom template. Falls back to DEFAULT_PROMPT_TEMPLATE.

    Returns:
        The formatted prompt string.
    """
    if template is None:
        template = DEFAULT_PROMPT_TEMPLATE
    return template.format(schema=schema, question=question)


def format_few_shot_prompt(
    schema: str,
    question: str,
    examples: list[dict[str, str]],
    template: str | None = None,
) -> str:
    """Format a few-shot prompt with in-context examples prepended.

    Args:
        schema: The database schema for the target question.
        question: The natural language question to answer.
        examples: List of dicts with keys 'schema', 'question', 'sql' for
                  in-context examples.
        template: Optional custom template for the final question.

    Returns:
        The full few-shot prompt string.
    """
    parts = [
        "Given the following database schema, write a SQL query that answers "
        "the question. Here are some examples:\n"
    ]
    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}\n")
        parts.append(
            FEW_SHOT_EXAMPLE_TEMPLATE.format(
                schema=ex["schema"],
                question=ex["question"],
                sql=ex["sql"],
            )
        )
    parts.append("### Your Turn\n")
    parts.append(f"Schema:\n{schema}\n")
    parts.append(f"Question: {question}\n")
    parts.append("SQL:")
    return "\n".join(parts)


# =============================================================================
# SQL Normalization (for exact-match comparison)
# =============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize a SQL query for exact-match comparison.

    Normalization steps:
    1. Parse and re-format with sqlparse (consistent whitespace, keyword casing)
    2. Uppercase all SQL keywords
    3. Strip trailing semicolons
    4. Collapse multiple whitespace to single spaces
    5. Strip leading/trailing whitespace

    This is intentionally aggressive — we want textually-different but
    logically-equivalent queries to still fail exact-match (that's what
    execution accuracy is for). The normalization only removes trivial
    formatting differences.

    Args:
        sql: Raw SQL query string.

    Returns:
        Normalized SQL string.
    """
    # Use sqlparse for initial formatting
    formatted = sqlparse.format(
        sql,
        keyword_case="upper",
        identifier_case="lower",
        strip_comments=True,
        reindent=False,
    )
    # Remove trailing semicolons
    formatted = formatted.rstrip(";").strip()
    # Collapse whitespace
    formatted = re.sub(r"\s+", " ", formatted)
    return formatted.strip()


# =============================================================================
# Constants
# =============================================================================

# Error analysis categories (used by eval scripts and error analysis helper)
ERROR_CATEGORIES = [
    "wrong_join",
    "wrong_aggregation",
    "schema_misunderstanding",
    "invalid_sql",
    "false_negative_exact_match",  # Textually different but semantically equivalent
    "other",
]
