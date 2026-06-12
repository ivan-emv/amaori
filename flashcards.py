import json
from dataclasses import dataclass
from typing import List

from .document_processor import DocumentKnowledgeBase
from .utils import extract_json_object, get_openai_client, safe_model_call


@dataclass
class Flashcard:
    front: str
    back: str
    reference: str


def generate_flashcards(kb: DocumentKnowledgeBase, model: str, total: int, level: str) -> List[Flashcard]:
    client = get_openai_client()
    if client is None:
        return []
    text = "\n\n".join([c["text"] for c in kb.chunks[:10]])[:30000]
    prompt = f"""
Genera {total} flashcards de estudio en español, nivel {level}, basadas solo en el documento.
Devuelve JSON: {{"flashcards":[{{"front":"pregunta o concepto","back":"respuesta breve","reference":"página/apartado"}}]}}
Documento:
{text}
""".strip()
    try:
        raw = safe_model_call(client, model, [{"role":"system","content":"Devuelve solo JSON válido."},{"role":"user","content":prompt}], temperature=0.25, json_mode=True)
        data = extract_json_object(raw)
        return [Flashcard(str(x.get("front","")), str(x.get("back","")), str(x.get("reference",""))) for x in data.get("flashcards", []) if x.get("front") and x.get("back")]
    except Exception:
        return []
