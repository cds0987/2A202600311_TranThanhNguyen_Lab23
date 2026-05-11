"""OpenAI-backed helpers for real LLM classification and evaluation."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@lru_cache(maxsize=1)
def _load_env_file() -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(dotenv_path=Path(".env"), override=False)


@lru_cache(maxsize=1)
def is_llm_enabled() -> bool:
    _load_env_file()
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    _load_env_file()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_model_name() -> str:
    _load_env_file()
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def parse_structured_output(
    *,
    instructions: str,
    user_input: str,
    text_format: type[SchemaT],
    temperature: float = 0.0,
) -> SchemaT:
    """Call the OpenAI Responses API and parse the result into a Pydantic schema."""
    response = _get_client().responses.parse(
        model=get_model_name(),
        instructions=instructions,
        input=user_input,
        text_format=text_format,
        temperature=temperature,
        timeout=20.0,
    )
    return response.output_parsed


def generate_text(
    *, instructions: str, user_input: str, temperature: float = 0.2, max_output_tokens: int = 200
) -> str:
    """Call the OpenAI Responses API and return plain text output."""
    response = _get_client().responses.create(
        model=get_model_name(),
        instructions=instructions,
        input=user_input,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=20.0,
    )
    return response.output_text.strip()
