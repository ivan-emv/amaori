from typing import List, Dict

from .document_processor import DocumentKnowledgeBase, search_relevant_chunks
from .utils import get_openai_client, safe_model_call


def answer_document_question(kb: DocumentKnowledgeBase, model: str, question: str, history: List[Dict[str, str]]) -> str:
    client = get_openai_client()
    if client is None:
        return "No se detectó OPENAI_API_KEY. Configure la clave para activar el chat con documento."
    chunks = search_relevant_chunks(kb, question, limit=5)
    context = "\n\n".join([c["text"] for c in chunks])[:30000]
    messages = [{"role":"system","content":"Responde únicamente con base en el contexto documental proporcionado. Si no está en el documento, indícalo con claridad. Cita páginas cuando sea posible."}]
    messages.extend(history[-6:])
    messages.append({"role":"user","content":f"Contexto documental:\n{context}\n\nPregunta:\n{question}"})
    return safe_model_call(client, model, messages, temperature=0.15)
