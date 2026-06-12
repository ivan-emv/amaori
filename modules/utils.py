import json
import os
import re
import random
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5"]
DEFAULT_MODEL = "gpt-4o-mini"


def get_secret_or_env(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def get_openai_client() -> Optional[Any]:
    api_key = get_secret_or_env("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def extract_json_object(raw: str) -> Dict[str, Any]:
    if not raw:
        raise ValueError("Respuesta vacía del modelo.")
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("No se encontró JSON válido en la respuesta del modelo.")
        return json.loads(match.group(0))


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_option_text(value: str, letter: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"^[a-dA-D][\)\.]\s*", "", value).strip()
    return f"{letter}) {value}"


def letter_to_index(letter: str) -> int:
    return {"a": 0, "b": 1, "c": 2, "d": 3}.get((letter or "").lower(), -1)


def index_to_letter(idx: int) -> str:
    return ["a", "b", "c", "d"][idx] if 0 <= idx <= 3 else "?"


def safe_model_call(client: Any, model: str, messages: List[Dict[str, str]], temperature: float = 0.2, json_mode: bool = False) -> str:
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs["model"] = DEFAULT_MODEL
        response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def estimate_auto_questions(total_chars: int, pages: int) -> int:
    base = max(10, min(200, total_chars // 2500))
    if pages <= 8:
        return min(max(base, 10), 20)
    if pages <= 40:
        return min(max(base, 20), 60)
    if pages <= 120:
        return min(max(base, 50), 120)
    return min(max(base, 80), 200)


def balanced_target_letters(n: int, seed: int) -> List[str]:
    letters = (["a", "b", "c", "d"] * ((n // 4) + 1))[:n]
    rnd = random.Random(seed)
    rnd.shuffle(letters)
    return letters
