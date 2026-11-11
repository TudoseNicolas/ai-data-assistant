"""Question -> pandas code -> run -> retry on error -> short explanation.

Only column names, types and a few sample rows are sent to the model;
the code itself runs locally on the full table.
"""
from __future__ import annotations

import builtins
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

try:  # plotly is only needed for charts
    import plotly.express as px
except ImportError:  # pragma: no cover
    px = None

DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
MAX_ATTEMPTS = 3

# model clients


class LLMClient(Protocol):
    """Anything with a name and a generate(prompt) -> text method can be used."""

    name: str

    def generate(self, prompt: str) -> str: ...


class ModelError(Exception):
    """The AI service itself failed (busy, wrong model name, bad key...), not the generated code."""


def _is_temporary(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).upper()
    return code in (429, 500, 502, 503, 504) or any(
        word in text for word in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "OVERLOADED", "HIGH DEMAND")
    )


def _is_quota(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 429 or "RESOURCE_EXHAUSTED" in str(exc).upper() or "QUOTA" in str(exc).upper()


def _is_missing_model(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 404 or "NOT_FOUND" in str(exc).upper()


class GeminiClient:
    """Google Gemini through the official google-genai SDK.

    If the chosen model is busy or no longer offered, it automatically switches to
    another "flash" model from the list Google returns for your key, and keeps using
    the one that worked.
    """

    name = "Gemini"
    retry_delays = (2, 5)  # seconds
    max_fallbacks = 3

    def __init__(self, api_key: str, model: str | None = None):
        from google import genai  # optional dependency

        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._client = genai.Client(api_key=api_key)
        self._fallbacks: list[str] | None = None

    def _fallback_models(self) -> list[str]:
        """Other text models to try: GEMINI_FALLBACK_MODELS from .env, else discovered from the API."""
        if self._fallbacks is None:
            configured = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
            if configured:
                self._fallbacks = configured
            else:
                skip = ("image", "tts", "audio", "live", "embedding", "vision", "robotics")
                found = []
                try:
                    for m in self._client.models.list():
                        name = (getattr(m, "name", "") or "").removeprefix("models/")
                        actions = getattr(m, "supported_actions", None) or []
                        if actions and "generateContent" not in actions:
                            continue
                        if "flash" in name and not any(word in name for word in skip):
                            found.append(name)
                except Exception:
                    found = []
                self._fallbacks = sorted(set(found), reverse=True)  # newest first
        return [m for m in self._fallbacks if m != self.model][: self.max_fallbacks]

    def _try_model(self, model: str, prompt: str) -> str:
        """Call one model, retrying a couple of times while it is busy."""
        for delay in (*self.retry_delays, None):
            try:
                response = self._client.models.generate_content(model=model, contents=prompt)
                return response.text or ""
            except Exception as exc:
                if not _is_temporary(exc) or delay is None:
                    raise
                time.sleep(delay)
        return ""

    def generate(self, prompt: str) -> str:
        tried: list[str] = []
        errors: list[Exception] = []
        for model in [self.model, *self._fallback_models()]:
            tried.append(model)
            try:
                text = self._try_model(model, prompt)
                self.model = model
                return text
            except Exception as exc:
                if not (_is_temporary(exc) or _is_missing_model(exc)):
                    raise ModelError(f"The AI service returned an error: {exc}") from exc
                errors.append(exc)  # busy / quota -> next model

        detail = str(errors[-1])[:300] if errors else ""
        models = ", ".join(tried)
        if any(_is_quota(e) for e in errors):
            message = (
                "Your Gemini key has reached its usage limit (quota) for the models tried "
                f"({models}). Free keys have per-minute and per-day limits; wait and try again, "
                "check your limits in Google AI Studio, or switch the AI engine to Local (Ollama)."
            )
        elif all(_is_missing_model(e) for e in errors):
            message = (
                f"None of these Gemini models are available to your key: {models}. "
                "Set GEMINI_MODEL in .env to a model listed in Google AI Studio."
            )
        else:
            message = (
                f"The Gemini models tried ({models}) are busy right now. "
                "Please try again in a minute, or switch the AI engine to Local (Ollama)."
            )
        raise ModelError(f"{message}\n\nGoogle said: {detail}") from (errors[-1] if errors else None)


DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaClient:
    """An open-source model running on your own computer through Ollama (https://ollama.com).

    No API key, no internet and no usage limits; the data never leaves the machine.
    """

    name = "Local"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 300):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        import json
        import urllib.request

        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"},
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode())

    def installed_models(self) -> list[str] | None:
        """Models downloaded in Ollama, or None if Ollama is not running."""
        try:
            return [m["name"] for m in self._request("/api/tags", timeout=1.5).get("models", [])]
        except Exception:
            return None

    def generate(self, prompt: str) -> str:
        import urllib.error

        try:
            reply = self._request("/api/generate", {
                "model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.1},
            })
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ModelError(
                    f"The local model “{self.model}” is not downloaded. "
                    f"Run `ollama pull {self.model}` in a terminal, then try again."
                ) from exc
            raise ModelError(f"Ollama returned an error: {exc}") from exc
        except Exception as exc:
            raise ModelError(
                "Can’t reach Ollama. Make sure the Ollama app is installed and running, then try again."
            ) from exc
        return reply.get("response", "")


# prompts

CODE_PROMPT = """You are a data analyst writing Python with pandas.

A pandas DataFrame called `df` is already loaded. Its columns and types:
{schema}

First rows:
{sample}

Question: {question}

Write Python code that answers the question.
Rules:
- Use only `df`, `pd` (pandas), `np` (numpy) and `px` (plotly.express). They are already imported; do not import anything.
- Put the answer in a variable called `result` (a DataFrame, Series, number or short string).
- If a chart helps, also create a plotly figure called `fig`, with a clear title.
- Do not modify files, the network or the system. Do not print.
- Reply with the code only, no explanation.
"""

FIX_PROMPT = """Your previous code failed.

Code:
{code}

Error:
{error}

Rewrite the code so it works. Same rules as before. Reply with the code only.
"""

EXPLAIN_PROMPT = """A user asked about their data: "{question}"

The analysis produced this result:
{result}

In one or two short sentences of plain English, tell the user what the result shows.
Mention the key numbers. Do not mention code, pandas or Python.
"""

# running code

_FORBIDDEN = [
    (re.compile(r"__"), "double underscores"),
    (re.compile(r"\bimport\b"), "import"),
    (re.compile(r"\b(open|exec|eval|compile|globals|locals|vars|getattr|setattr|delattr|input|breakpoint)\s*\("), "a system function"),
    (re.compile(r"\b(os|sys|subprocess|shutil|pathlib|socket|requests|urllib)\b"), "a system module"),
    (re.compile(r"\.to_(csv|excel|pickle|parquet|sql|json|html|clipboard)\s*\("), "writing files"),
    (re.compile(r"\bpd\.read_\w+"), "reading files"),
]
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "isinstance", "len", "list", "map", "max", "min", "range", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
    ]
}
_ALLOWED_IMPORT = re.compile(
    r"^\s*(import\s+(pandas|numpy|plotly\.express)(\s+as\s+\w+)?|from\s+plotly\s+import\s+express(\s+as\s+\w+)?)\s*$"
)


class UnsafeCodeError(Exception):
    """The generated code tried to do something outside data analysis."""


def extract_code(text: str) -> str:
    """Take the code out of a model reply, with or without ``` fences."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    # pd/np/px are already in the namespace, drop the imports
    lines = [line for line in code.strip().splitlines() if not _ALLOWED_IMPORT.match(line)]
    return "\n".join(lines).strip()


def check_code_safety(code: str) -> None:
    """Basic checks before exec(). Not a real sandbox."""
    for pattern, what in _FORBIDDEN:
        if pattern.search(code):
            raise UnsafeCodeError(f"The generated code tried to use {what}, which is not allowed.")


def run_code(code: str, df: pd.DataFrame) -> tuple[Any, Any]:
    """Run generated code on a copy of the data and return (result, fig)."""
    check_code_safety(code)
    namespace = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "px": px, "df": df.copy()}
    exec(code, namespace)  # noqa: S102 - restricted namespace, see check_code_safety
    result, fig = namespace.get("result"), namespace.get("fig")
    if result is None and fig is None:
        raise ValueError("The code did not create `result` or `fig`.")
    return result, fig


# the agent


@dataclass
class AgentAnswer:
    question: str
    code: str = ""
    result: Any = None
    fig: Any = None
    explanation: str = ""
    attempts: int = 0
    error: str | None = None
    model: str = ""


def describe_schema(df: pd.DataFrame) -> str:
    return "\n".join(f"- {col} ({dtype})" for col, dtype in df.dtypes.astype(str).items())


def result_to_text(result: Any, max_rows: int = 15) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.head(max_rows).to_string()
    return str(result)[:2000]


def ask(df: pd.DataFrame, question: str, llm: LLMClient, max_attempts: int = MAX_ATTEMPTS) -> AgentAnswer:
    """Answer a question about df: generate code, run it, self-correct on errors."""
    answer = AgentAnswer(question=question)
    prompt = CODE_PROMPT.format(
        schema=describe_schema(df),
        sample=df.head(5).to_string(max_cols=30),
        question=question,
    )
    for attempt in range(1, max_attempts + 1):
        answer.attempts = attempt
        try:
            reply = llm.generate(prompt)
        except Exception as exc:  # API error, retrying the code won't help
            answer.error = str(exc) if isinstance(exc, ModelError) else f"The AI service returned an error: {exc}"
            break
        try:
            answer.code = extract_code(reply)
            answer.result, answer.fig = run_code(answer.code, df)
            answer.error = None
            break
        except UnsafeCodeError as exc:
            answer.error = str(exc)
            break
        except Exception as exc:
            answer.error = f"{type(exc).__name__}: {exc}"
            prompt = prompt + "\n\n" + FIX_PROMPT.format(code=answer.code, error=answer.error)

    if answer.error is None:
        answer.model = getattr(llm, "model", "")
        shown = answer.result if answer.result is not None else "(a chart)"
        try:
            answer.explanation = llm.generate(
                EXPLAIN_PROMPT.format(question=question, result=result_to_text(shown))
            ).strip()
        except Exception:
            answer.explanation = ""
    return answer


# suggestions


def suggest_questions(df: pd.DataFrame, limit: int = 4) -> list[str]:
    """Example questions built from the columns, without calling the model."""
    def looks_like_id(col: str) -> bool:
        return bool(re.search(r"(^|[\s_])id$|^id[\s_]|\bcode\b", col.lower()))

    numeric = [c for c in df.select_dtypes("number").columns if not looks_like_id(c)]
    categorical = [
        c for c in df.columns
        if c not in numeric and not looks_like_id(c)
        and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
        and 2 <= df[c].nunique() <= 50
    ]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dates:
        dates = [c for c in df.columns if "date" in c.lower()]

    # main numeric column: prefer money-like names, else the largest values
    keywords = ["revenue", "sales", "total", "amount", "income", "salary", "value", "cost", "price"]
    money = [(rank, c) for c in numeric for rank, k in enumerate(keywords) if k in c.lower()]
    main = min(money)[1] if money else (max(numeric, key=lambda c: df[c].abs().mean()) if numeric else None)
    categorical.sort(key=lambda c: df[c].nunique())

    questions = []
    if categorical and main:
        questions.append(f"Top 5 {categorical[0]} by total {main}")
    if dates and main:
        questions.append(f"{main} over time, by month")
    if len(categorical) > 1:
        questions.append(f"How many rows per {categorical[1]}?")
    elif categorical:
        questions.append(f"How many rows per {categorical[0]}?")
    if len(categorical) > 1 and main:
        questions.append(f"Average {main} per {categorical[1]}")
    if main:
        questions.append(f"Distribution of {main}")
    return questions[:limit]
