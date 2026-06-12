import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .document_processor import DocumentKnowledgeBase
from .utils import clean_option_text, extract_json_object, get_openai_client, letter_to_index, safe_model_call, balanced_target_letters


@dataclass
class GeneratedQuestion:
    qnum: int
    text: str
    options: List[str]
    correct_letter: str
    explanation: str
    reference: str
    page: int
    source_excerpt: str = ""


def validate_generated_question(item: Dict[str, Any], qnum: int, fallback_page: int) -> Optional[GeneratedQuestion]:
    question = str(item.get("pregunta") or item.get("question") or "").strip()
    options_raw = item.get("opciones") or item.get("options") or {}
    correct_letter = str(item.get("respuesta_correcta") or item.get("correct_letter") or "").strip().lower()
    explanation = str(item.get("explicacion") or item.get("explanation") or "").strip()
    reference = str(item.get("referencia") or item.get("reference") or "").strip()
    source_excerpt = str(item.get("fragmento_base") or item.get("source_excerpt") or "").strip()
    page = item.get("pagina") or item.get("page") or fallback_page
    if not question:
        return None
    letters = ["a", "b", "c", "d"]
    if isinstance(options_raw, dict):
        options = [clean_option_text(options_raw.get(letter, ""), letter) for letter in letters]
    elif isinstance(options_raw, list):
        options = [clean_option_text(options_raw[i] if i < len(options_raw) else "", letters[i]) for i in range(4)]
    else:
        return None
    if len(options) != 4 or any(len(opt) <= 3 for opt in options):
        return None
    if correct_letter not in set(letters):
        return None
    try:
        page_int = int(page)
    except Exception:
        page_int = int(fallback_page)
    return GeneratedQuestion(qnum, question, options, correct_letter, explanation or "Explicación no disponible.", reference or f"Página {page_int}", page_int, source_excerpt)


def force_correct_letter(q: GeneratedQuestion, desired_letter: str, seed: int) -> GeneratedQuestion:
    correct_idx = letter_to_index(q.correct_letter)
    desired_idx = letter_to_index(desired_letter)
    if correct_idx < 0 or desired_idx < 0:
        return q
    option_texts = [re.sub(r"^[a-dA-D]\)\s*", "", str(opt)).strip() for opt in q.options]
    remaining = [i for i in range(4) if i != correct_idx]
    random.Random(seed).shuffle(remaining)
    order = [None, None, None, None]
    order[desired_idx] = correct_idx
    for pos in range(4):
        if order[pos] is None:
            order[pos] = remaining.pop(0)
    letters = ["a", "b", "c", "d"]
    return GeneratedQuestion(
        q.qnum,
        q.text,
        [clean_option_text(option_texts[old_idx], letters[new_idx]) for new_idx, old_idx in enumerate(order)],
        desired_letter,
        q.explanation,
        q.reference,
        q.page,
        q.source_excerpt,
    )


def balance_questions(questions: List[GeneratedQuestion], seed: int) -> List[GeneratedQuestion]:
    targets = balanced_target_letters(len(questions), seed)
    return [force_correct_letter(q, targets[i], seed + i) for i, q in enumerate(questions)]


def generate_questions(kb: DocumentKnowledgeBase, model: str, total_questions: int, difficulty: str, doc_type: str, questions_per_chunk: int = 4) -> List[GeneratedQuestion]:
    client = get_openai_client()
    if client is None:
        return heuristic_questions(kb, total_questions)
    system = "Eres un experto en formación, evaluación y diseño de preguntas tipo test. Devuelve exclusivamente JSON válido."
    all_questions: List[GeneratedQuestion] = []
    qnum = 1
    remaining = total_questions
    for chunk in kb.chunks:
        if remaining <= 0:
            break
        n = min(questions_per_chunk, remaining)
        pages_label = ", ".join(str(p) for p in chunk.get("pages", []))
        prompt = f"""
Documento: {doc_type}
Dificultad: {difficulty}
Páginas: {pages_label}
Número de preguntas: {n}

Genera exactamente {n} preguntas tipo test basadas solo en el fragmento.
Reglas:
- 4 opciones: a, b, c, d.
- Una sola correcta.
- Distractores verosímiles pero falsos.
- No reveles la respuesta en la referencia.
- Evita que la respuesta correcta esté siempre en a.
- Incluye explicación breve, página y fragmento base.
- No inventes información.

JSON obligatorio:
{{"preguntas":[{{"pregunta":"...","opciones":{{"a":"...","b":"...","c":"...","d":"..."}},"respuesta_correcta":"a","explicacion":"...","referencia":"Página X / apartado si aplica","pagina":1,"fragmento_base":"..."}}]}}

Fragmento:
{chunk['text']}
""".strip()
        try:
            raw = safe_model_call(client, model, [{"role":"system","content":system},{"role":"user","content":prompt}], temperature=0.25, json_mode=True)
            data = extract_json_object(raw)
            items = data.get("preguntas", []) if isinstance(data, dict) else []
        except Exception:
            items = []
        fallback_page = int(chunk.get("pages", [1])[0]) if chunk.get("pages") else 1
        for item in items:
            q = validate_generated_question(item, qnum, fallback_page)
            if q:
                all_questions.append(q)
                qnum += 1
                remaining -= 1
                if remaining <= 0:
                    break
    return balance_questions(all_questions[:total_questions], seed=abs(hash((kb.source_label, total_questions, difficulty))) % 1000000)


def heuristic_questions(kb: DocumentKnowledgeBase, total_questions: int) -> List[GeneratedQuestion]:
    candidates = []
    for rec in kb.pages:
        sentences = re.split(r"(?<=[\.\?\!])\s+", re.sub(r"\s+", " ", rec.text))
        for s in sentences:
            if 120 <= len(s.strip()) <= 420:
                candidates.append((rec.page, s.strip()))
    random.Random(42).shuffle(candidates)
    questions = []
    for idx, (page, sent) in enumerate(candidates[:total_questions], start=1):
        questions.append(GeneratedQuestion(
            idx,
            "¿Cuál de las siguientes afirmaciones se corresponde con el documento?",
            [clean_option_text(sent[:260], "a"), clean_option_text("El documento establece expresamente lo contrario.", "b"), clean_option_text("La materia no aparece mencionada en el documento.", "c"), clean_option_text("La afirmación solo aplica fuera del documento analizado.", "d")],
            "a",
            "Pregunta generada en modo básico por falta de conexión IA. Revise la referencia.",
            f"Página {page}", page, sent[:260]
        ))
    return balance_questions(questions, seed=99001)
