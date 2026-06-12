import math
import random
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.concept_maps import generate_concept_map
from modules.document_chat import answer_document_question
from modules.document_processor import build_knowledge_base, download_drive_file, extract_drive_file_id
from modules.flashcards import generate_flashcards
from modules.question_bank import GeneratedQuestion, generate_questions
from modules.study_material import generate_study_material
from modules.utils import AVAILABLE_MODELS, DEFAULT_MODEL, estimate_auto_questions, get_secret_or_env, index_to_letter, letter_to_index

st.set_page_config(page_title="Estudio Inteligente IA", page_icon="📚", layout="wide")

# Configuración interna: se oculta al usuario final para simplificar la experiencia.
INTERNAL_MAX_CHARS = 10000


def init_state():
    defaults = {
        "file_bytes": None,
        "filename": "",
        "source_label": "",
        "kb": None,
        "study_material": None,
        "questions": [],
        "quiz": [],
        "quiz_answers": {},
        "flashcards": [],
        "flashcard_idx": 0,
        "chat_history": [],
        "concept_map": "",
        "seed": random.randint(1, 10_000_000),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_document():
    for key in ["file_bytes", "filename", "source_label", "kb", "study_material", "questions", "quiz", "quiz_answers", "flashcards", "chat_history", "concept_map"]:
        st.session_state[key] = [] if key in {"questions", "quiz", "flashcards", "chat_history"} else None
    st.session_state.filename = ""
    st.session_state.source_label = ""
    st.session_state.flashcard_idx = 0
    st.session_state.seed = random.randint(1, 10_000_000)


def render_mermaid(code: str):
    html = f"""
    <div class="mermaid">{code}</div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    components.html(html, height=650, scrolling=True)


def export_questions_csv(questions: List[GeneratedQuestion]) -> bytes:
    rows = []
    for q in questions:
        rows.append({
            "Nº": q.qnum,
            "Pregunta": q.text,
            "A": q.options[0][3:] if len(q.options) > 0 else "",
            "B": q.options[1][3:] if len(q.options) > 1 else "",
            "C": q.options[2][3:] if len(q.options) > 2 else "",
            "D": q.options[3][3:] if len(q.options) > 3 else "",
            "Correcta": q.correct_letter,
            "Explicación": q.explanation,
            "Referencia": q.reference,
            "Página": q.page,
            "Fragmento base": q.source_excerpt,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def sidebar_config():
    with st.sidebar:
        st.markdown("## 📚 Estudio Inteligente IA")
        st.caption("Carga un documento y genera material de estudio, test, flashcards, mapas y chat contextual.")
        st.divider()

        st.header("📄 Documento")
        uploaded = st.file_uploader("Subir archivo", type=["pdf", "docx", "txt"], accept_multiple_files=False)
        drive_url = st.text_input("Enlace Google Drive", value="")
        col1, col2 = st.columns(2)
        with col1:
            load_drive = st.button("📥 Drive", use_container_width=True)
        with col2:
            clear = st.button("🧹 Limpiar", use_container_width=True)

        st.divider()
        st.header("🤖 IA")
        model = st.selectbox("Modelo", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))
        level = st.selectbox("Nivel", ["Básico", "Intermedio", "Avanzado", "Experto"], index=2)
        mode = st.selectbox("Modo de procesamiento", ["Automático", "Cuestionario existente", "Generar contenido IA"], index=0)
        st.divider()
        st.header("📝 Preguntas")
        q_mode = st.radio("Cantidad", ["Automático", "Personalizado"], horizontal=True)
        total_q = 30
        if q_mode == "Personalizado":
            total_q = st.slider("Preguntas a generar", 5, 200, 30, 5)
        quiz_size = st.slider("Preguntas por test", 5, 100, 20, 5)

        if clear:
            clear_document()
            st.rerun()

        if uploaded is not None:
            st.session_state.file_bytes = uploaded.read()
            st.session_state.filename = uploaded.name
            st.session_state.source_label = f"Archivo subido: {uploaded.name}"

        if load_drive:
            fid = extract_drive_file_id(drive_url)
            if not fid:
                st.error("No se pudo extraer el ID de Drive.")
            else:
                with st.spinner("Descargando archivo desde Drive..."):
                    try:
                        st.session_state.file_bytes = download_drive_file(fid)
                        st.session_state.filename = "documento.pdf"
                        st.session_state.source_label = f"Google Drive file_id: {fid}"
                        st.success("Documento cargado.")
                    except Exception as e:
                        st.error(f"No se pudo descargar: {e}")

        return model, level, mode, q_mode, total_q, quiz_size


def page_document(model, level, mode, q_mode, total_q, quiz_size):
    st.title("📚 Estudio Inteligente IA")

    if not get_secret_or_env("OPENAI_API_KEY", ""):
        st.warning("No se detectó OPENAI_API_KEY. Algunas funciones usarán respaldo básico o quedarán desactivadas.")

    if st.session_state.file_bytes is None:
        st.info("Sube un PDF, DOCX o TXT para comenzar.")
        st.stop()

    if st.session_state.kb is None:
        with st.spinner("Procesando documento y creando base de conocimiento temporal..."):
            try:
                st.session_state.kb = build_knowledge_base(
                    st.session_state.file_bytes,
                    st.session_state.filename,
                    st.session_state.source_label,
                    max_chars=INTERNAL_MAX_CHARS,
                )
            except Exception as e:
                st.error(f"No se pudo procesar el documento: {e}")
                st.stop()

    kb = st.session_state.kb
    total_chars = len(kb.full_text)
    auto_q = estimate_auto_questions(total_chars, len(kb.pages))
    selected_total_q = auto_q if q_mode == "Automático" else int(total_q)
    auto_questions_per_chunk = max(2, min(8, math.ceil(selected_total_q / max(1, len(kb.chunks)))))

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Páginas", len(kb.pages))
    col_b.metric("Bloques", len(kb.chunks))
    col_c.metric("Caracteres", f"{total_chars:,}".replace(",", "."))
    col_d.metric("Preguntas sugeridas", auto_q)

    st.info(f"Tipo detectado: **{kb.detected_type}** | Confianza: **{kb.detection_confidence}** | {kb.detection_notes}")
    if mode != "Automático":
        st.caption(f"Modo manual seleccionado: {mode}")

    tabs = st.tabs(["📖 Temario", "📝 Banco de preguntas", "🎯 Test", "🃏 Flashcards", "💬 Chat", "🧠 Mapa conceptual", "📄 Documento"])

    with tabs[0]:
        st.subheader("📖 Temario y resumen")
        if st.button("Generar temario", type="primary", use_container_width=True):
            with st.spinner("Generando resumen, temario y conceptos clave..."):
                st.session_state.study_material = generate_study_material(kb, model, level)
        if st.session_state.study_material:
            material = st.session_state.study_material
            st.markdown("### Resumen ejecutivo")
            st.markdown(material.get("resumen", ""))
            st.markdown("### Temario estructurado")
            st.markdown(material.get("temario", ""))
            st.markdown("### Conceptos clave")
            st.markdown(material.get("conceptos", ""))
        else:
            st.info("Pulsa el botón para generar el temario del documento.")

    with tabs[1]:
        st.subheader("📝 Banco de preguntas")
        if st.button("Generar banco de preguntas", type="primary", use_container_width=True):
            with st.spinner("Generando preguntas y equilibrando respuestas correctas..."):
                st.session_state.questions = generate_questions(kb, model, int(selected_total_q), level, kb.detected_type, int(auto_questions_per_chunk))
                st.session_state.quiz = []
                st.session_state.quiz_answers = {}
        questions = st.session_state.questions
        if questions:
            st.success(f"Banco generado: {len(questions)} preguntas")
            st.download_button("⬇️ Descargar CSV", data=export_questions_csv(questions), file_name="banco_preguntas.csv", mime="text/csv")
            for idx, q in enumerate(questions, start=1):
                with st.expander(f"{q.qnum}. {q.text[:100]}...", expanded=False):
                    st.markdown(f"**Pregunta:** {q.text}")
                    for opt in q.options:
                        st.write(opt)
                    st.caption(f"Referencia de estudio: Página {q.page}")
                    show_key = f"show_answer_{idx}"
                    if show_key not in st.session_state:
                        st.session_state[show_key] = False
                    if not st.session_state[show_key]:
                        if st.button("👁️ Ver respuesta", key=f"show_btn_{idx}"):
                            st.session_state[show_key] = True
                            st.rerun()
                    else:
                        if st.button("🙈 Ocultar respuesta", key=f"hide_btn_{idx}"):
                            st.session_state[show_key] = False
                            st.rerun()
                        st.success(f"Correcta: {q.correct_letter})")
                        st.info(q.explanation)
                        st.caption(f"Referencia: {q.reference}")
                        if q.source_excerpt:
                            st.caption(f"Base documental: {q.source_excerpt}")
        else:
            st.info("Genera el banco de preguntas para crear tests y exportar contenido.")

    with tabs[2]:
        st.subheader("🎯 Test")
        if not st.session_state.questions:
            st.info("Primero genera un banco de preguntas.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Crear test", type="primary", use_container_width=True):
                    pool = st.session_state.questions[:]
                    random.shuffle(pool)
                    st.session_state.quiz = pool[: min(len(pool), int(quiz_size))]
                    st.session_state.quiz_answers = {}
                    st.rerun()
            with col2:
                if st.button("Reiniciar respuestas", use_container_width=True):
                    st.session_state.quiz_answers = {}
                    st.rerun()

            if st.session_state.quiz:
                for i, q in enumerate(st.session_state.quiz, start=1):
                    key = f"quiz_q_{i}"
                    st.markdown(f"**{i}.** {q.text}")
                    prev = st.session_state.quiz_answers.get(key)
                    idx = 0 if prev is None else prev + 1
                    choice = st.radio("", ["— Seleccione una opción —"] + q.options, index=idx, key=f"radio_{i}", label_visibility="collapsed")
                    if choice == "— Seleccione una opción —":
                        st.session_state.quiz_answers[key] = None
                    else:
                        st.session_state.quiz_answers[key] = letter_to_index(choice.split(")")[0].strip().lower())
                    st.caption(f"Referencia de estudio: Página {q.page}")
                    st.divider()

                if st.button("Calcular resultado", type="primary", use_container_width=True):
                    total = len(st.session_state.quiz)
                    answered = correct = incorrect = blank = 0
                    results = []
                    for i, q in enumerate(st.session_state.quiz, start=1):
                        user_idx = st.session_state.quiz_answers.get(f"quiz_q_{i}")
                        correct_idx = letter_to_index(q.correct_letter)
                        if user_idx is None:
                            blank += 1
                        elif user_idx == correct_idx:
                            answered += 1; correct += 1
                        else:
                            answered += 1; incorrect += 1
                        results.append((i, q, user_idx, correct_idx))
                    score = round((correct / total) * 10, 2) if total else 0
                    st.info(f"Total: {total} | Respondidas: {answered} | Aciertos: {correct} | Errores: {incorrect} | Sin responder: {blank} | Nota: {score}/10")
                    with st.expander("Revisión detallada", expanded=True):
                        for i, q, user_idx, correct_idx in results:
                            user_letter = "—" if user_idx is None else index_to_letter(user_idx)
                            corr_letter = index_to_letter(correct_idx)
                            if user_idx is not None and user_idx == correct_idx:
                                st.success(f"{i}) Correcta ✅ | Tu respuesta: {user_letter} | Correcta: {corr_letter}")
                            else:
                                st.error(f"{i}) Incorrecta ❌ | Tu respuesta: {user_letter} | Correcta: {corr_letter}")
                            st.markdown(f"**Explicación:** {q.explanation}")
                            st.caption(f"Referencia: {q.reference}")
                            if q.source_excerpt:
                                st.caption(f"Base documental: {q.source_excerpt}")
                            st.markdown("---")
            else:
                st.info("Pulsa 'Crear test' para comenzar.")

    with tabs[3]:
        st.subheader("🃏 Flashcards")
        flash_total = st.slider("Número de flashcards", 5, 80, 20, 5)
        if st.button("Generar flashcards", type="primary", use_container_width=True):
            with st.spinner("Generando flashcards..."):
                st.session_state.flashcards = generate_flashcards(kb, model, int(flash_total), level)
                st.session_state.flashcard_idx = 0
        cards = st.session_state.flashcards
        if cards:
            idx = min(st.session_state.flashcard_idx, len(cards) - 1)
            card = cards[idx]
            st.markdown(f"### Tarjeta {idx + 1}/{len(cards)}")
            st.markdown(f"**Anverso:** {card.front}")
            with st.expander("Mostrar reverso", expanded=False):
                st.markdown(card.back)
                st.caption(card.reference)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("⬅️ Anterior"):
                    st.session_state.flashcard_idx = max(0, idx - 1)
                    st.rerun()
            with c2:
                if st.button("Siguiente ➡️"):
                    st.session_state.flashcard_idx = min(len(cards) - 1, idx + 1)
                    st.rerun()
        else:
            st.info("Genera flashcards desde el documento.")

    with tabs[4]:
        st.subheader("💬 Chat con el documento")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        prompt = st.chat_input("Haz una pregunta sobre el documento...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Consultando el documento..."):
                    answer = answer_document_question(kb, model, prompt, st.session_state.chat_history[:-1])
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with tabs[5]:
        st.subheader("🧠 Mapa conceptual")
        if st.button("Generar mapa conceptual", type="primary", use_container_width=True):
            with st.spinner("Generando mapa conceptual..."):
                st.session_state.concept_map = generate_concept_map(kb, model)
        if st.session_state.concept_map:
            render_mermaid(st.session_state.concept_map)
            st.code(st.session_state.concept_map, language="mermaid")
        else:
            st.info("Genera un mapa conceptual del documento.")

    with tabs[6]:
        st.subheader("📄 Documento procesado")
        st.write(kb.source_label)
        preview = kb.full_text[:12000]
        st.text_area("Texto extraído", preview, height=500)
        if not preview.strip():
            st.warning("No se extrajo texto. El documento podría requerir OCR.")


def main():
    init_state()
    config = sidebar_config()
    page_document(*config)


if __name__ == "__main__":
    main()
