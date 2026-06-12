import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

try:
    import fitz
except Exception:
    fitz = None

try:
    import docx
except Exception:
    docx = None

from .utils import normalize_text


@dataclass
class PageRecord:
    page: int
    text: str


@dataclass
class DocumentKnowledgeBase:
    source_label: str
    file_type: str
    pages: List[PageRecord]
    full_text: str
    chunks: List[Dict[str, Any]]
    detected_type: str
    detection_confidence: str
    detection_notes: str


SOLUTIONS_HEADER_RE = re.compile(r"(?im)^\s*Soluciones\s*$")
QUESTION_RE = re.compile(r"(?m)^\s*\d+\.\s+.+")
OPTION_RE = re.compile(r"(?m)^\s*[a-dA-D]\)\s+.+")
SOLUTION_PAIR_RE = re.compile(r"(?i)\b(\d+)\.\s*([a-d])\b")


def extract_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_drive_file(file_id: str, timeout: int = 60) -> bytes:
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True, timeout=timeout)
    response.raise_for_status()
    if "content-disposition" in response.headers:
        return response.content
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if not token:
        match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
        if match:
            token = match.group(1)
    if not token:
        raise RuntimeError("Google Drive no permite descarga directa. Comparta el archivo como 'Cualquiera con el enlace'.")
    response2 = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=timeout)
    response2.raise_for_status()
    return response2.content


@st.cache_data(show_spinner=False)
def extract_pdf_pages(file_bytes: bytes) -> List[PageRecord]:
    if fitz is None:
        raise RuntimeError("Falta PyMuPDF. Instale pymupdf en requirements.txt.")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    records: List[PageRecord] = []
    for idx in range(doc.page_count):
        text = doc.load_page(idx).get_text("text") or ""
        records.append(PageRecord(page=idx + 1, text=normalize_text(text)))
    doc.close()
    return records


@st.cache_data(show_spinner=False)
def extract_docx_pages(file_bytes: bytes) -> List[PageRecord]:
    if docx is None:
        raise RuntimeError("Falta python-docx en requirements.txt.")
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    text = normalize_text("\n\n".join(paragraphs))
    return [PageRecord(page=1, text=text)] if text else []


@st.cache_data(show_spinner=False)
def extract_txt_pages(file_bytes: bytes) -> List[PageRecord]:
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            text = file_bytes.decode(enc)
            break
        except Exception:
            text = ""
    text = normalize_text(text)
    return [PageRecord(page=1, text=text)] if text else []


def extract_pages(file_bytes: bytes, filename: str) -> List[PageRecord]:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return extract_pdf_pages(file_bytes)
    if ext == "docx":
        return extract_docx_pages(file_bytes)
    if ext == "txt":
        return extract_txt_pages(file_bytes)
    raise RuntimeError("Formato no soportado. Use PDF, DOCX o TXT.")


def build_chunks(pages: List[PageRecord], max_chars: int = 6500) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_pages: List[int] = []
    current_len = 0
    for rec in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", rec.text) if p.strip()] or [rec.text]
        for paragraph in paragraphs:
            piece = f"[Página {rec.page}]\n{paragraph}"
            if current_parts and current_len + len(piece) > max_chars:
                chunks.append({"pages": current_pages[:], "text": "\n\n".join(current_parts)})
                current_parts, current_pages, current_len = [], [], 0
            current_parts.append(piece)
            if rec.page not in current_pages:
                current_pages.append(rec.page)
            current_len += len(piece)
    if current_parts:
        chunks.append({"pages": current_pages[:], "text": "\n\n".join(current_parts)})
    return chunks


def detect_document_type(full_text: str) -> Dict[str, str]:
    sample = full_text[:50000]
    q_count = len(QUESTION_RE.findall(sample))
    opt_count = len(OPTION_RE.findall(sample))
    sol_header = bool(SOLUTIONS_HEADER_RE.search(sample))
    sol_pairs = len(SOLUTION_PAIR_RE.findall(sample))
    if q_count >= 5 and opt_count >= 20 and sol_header and sol_pairs >= 5:
        return {"type": "Cuestionario existente", "confidence": "Alta", "notes": "Se detectaron preguntas, opciones y bloque de soluciones."}
    if q_count >= 5 and opt_count >= 20:
        return {"type": "Cuestionario parcial", "confidence": "Media", "notes": "Se detectaron preguntas y opciones, pero no soluciones claras."}
    if len(sample.strip()) < 300:
        return {"type": "Documento sin texto/OCR requerido", "confidence": "Alta", "notes": "El texto extraído es insuficiente."}
    return {"type": "Documento de estudio", "confidence": "Alta", "notes": "No se detectó formato de test; se recomienda generar material con IA."}


def build_knowledge_base(file_bytes: bytes, filename: str, source_label: str, max_chars: int = 6500) -> DocumentKnowledgeBase:
    pages = extract_pages(file_bytes, filename)
    full_text = normalize_text("\n\n".join([f"[Página {p.page}]\n{p.text}" for p in pages if p.text]))
    detection = detect_document_type(full_text)
    return DocumentKnowledgeBase(
        source_label=source_label,
        file_type=filename.lower().split(".")[-1],
        pages=pages,
        full_text=full_text,
        chunks=build_chunks(pages, max_chars=max_chars),
        detected_type=detection["type"],
        detection_confidence=detection["confidence"],
        detection_notes=detection["notes"],
    )


def search_relevant_chunks(kb: DocumentKnowledgeBase, query: str, limit: int = 4) -> List[Dict[str, Any]]:
    words = [w.lower() for w in re.findall(r"\w{4,}", query or "")]
    if not words:
        return kb.chunks[:limit]
    scored = []
    for chunk in kb.chunks:
        text_lower = chunk["text"].lower()
        score = sum(text_lower.count(w) for w in words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]] or kb.chunks[:limit]
