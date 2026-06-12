from .document_processor import DocumentKnowledgeBase
from .utils import get_openai_client, safe_model_call


def generate_concept_map(kb: DocumentKnowledgeBase, model: str) -> str:
    client = get_openai_client()
    if client is None:
        return fallback_mermaid(kb)
    text = kb.full_text[:24000]
    prompt = f"""
Genera un mapa conceptual en sintaxis Mermaid mindmap basado solo en el documento.
No uses bloques markdown; devuelve únicamente el código Mermaid empezando por 'mindmap'.
Documento:
{text}
""".strip()
    raw = safe_model_call(client, model, [{"role":"system","content":"Devuelve solo Mermaid válido."},{"role":"user","content":prompt}], temperature=0.2)
    raw = raw.replace("```mermaid", "").replace("```", "").strip()
    return raw if raw.startswith("mindmap") else fallback_mermaid(kb)


def fallback_mermaid(kb: DocumentKnowledgeBase) -> str:
    lines = ["mindmap", "  root((Documento))"]
    for i, chunk in enumerate(kb.chunks[:8], start=1):
        page = chunk.get("pages", ["?"])[0]
        label = chunk.get("text", "Tema").split("\n")[-1][:45].replace(":", " ")
        lines.append(f"    Página {page}")
        lines.append(f"      {label}")
    return "\n".join(lines)
