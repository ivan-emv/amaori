from typing import Dict

from .document_processor import DocumentKnowledgeBase
from .utils import get_openai_client, safe_model_call


def generate_study_material(kb: DocumentKnowledgeBase, model: str, level: str) -> Dict[str, str]:
    client = get_openai_client()
    sample = kb.full_text[:28000]
    if client is None:
        return {
            "resumen": "No se detectó OPENAI_API_KEY. Configure la clave para generar el temario con IA.",
            "temario": fallback_outline(kb),
            "conceptos": "Configure OPENAI_API_KEY para generar conceptos clave.",
        }
    system = "Eres un especialista en creación de material de estudio claro, estructurado y fiel al documento."
    prompt = f"""
Nivel de profundidad: {level}

A partir del documento, genera en español:
1) Resumen ejecutivo.
2) Temario estructurado por temas y subtemas.
3) Conceptos clave, plazos, obligaciones, definiciones o ideas esenciales.

No inventes información y mantén referencias generales a páginas o apartados cuando sea posible.

Documento:
{sample}
""".strip()
    raw = safe_model_call(client, model, [{"role":"system","content":system},{"role":"user","content":prompt}], temperature=0.2)
    return split_material(raw)


def split_material(raw: str) -> Dict[str, str]:
    text = raw or ""
    return {"resumen": text, "temario": text, "conceptos": text}


def fallback_outline(kb: DocumentKnowledgeBase) -> str:
    lines = []
    for chunk in kb.chunks[:8]:
        page = chunk.get("pages", ["?"])[0]
        first = chunk.get("text", "").replace("\n", " ")[:280]
        lines.append(f"- Página {page}: {first}...")
    return "\n".join(lines) if lines else "No hay texto suficiente."
