"""
app.py - LegalLens
"""

import json
from html import escape
import os
import sys

# Ensure article_src/src modules are discovered
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "article_src", "src"))

import pandas as pd
from PIL import Image
import streamlit as st
import torch

from ui_formatting import _escape_html, _md_to_html
from multilingual_judgebert import load_checkpoint, predict_single_pair
from legal_simplifier import (
    generate_simplified_clause,
    extract_text_from_document_image,
    find_matching_clauses,
)
from doc_chat import (
    extract_text_from_document,
    chunk_text,
    build_vector_index,
    chat_with_document,
    validate_uploaded_file,
    format_file_size,
)

# ---------------------------------------------------------------------------
# Premium Chat UI CSS
# ---------------------------------------------------------------------------
CHAT_CSS = """
<style>
/* ── Chat page layout ──────────────────────────────────────────────── */
.chat-page-wrapper { display: flex; flex-direction: column; height: 100%; }

/* ── Document info card ─────────────────────────────────────────────── */
.doc-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #232b3e 100%);
    border: 1px solid #2d3a52;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.doc-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b82a8;
    margin-bottom: 10px;
}
.doc-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    word-break: break-all;
    margin-bottom: 4px;
}
.doc-meta {
    font-size: 0.78rem;
    color: #6b82a8;
    margin-bottom: 2px;
}
.doc-status-ok  { color: #4ade80; font-size: 0.8rem; font-weight: 600; }
.doc-status-err { color: #f87171; font-size: 0.8rem; font-weight: 600; }

/* ── Empty state ────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #4a5568;
}
.empty-state-icon { font-size: 3.5rem; margin-bottom: 16px; }
.empty-state-title { font-size: 1.3rem; font-weight: 700; color: #718096; margin-bottom: 8px; }
.empty-state-sub   { font-size: 0.9rem; color: #4a5568; line-height: 1.6; }

/* ── Chat container ─────────────────────────────────────────────────── */
.chat-container {
    background: #0f1623;
    border: 1px solid #1e2a3a;
    border-radius: 16px;
    padding: 20px 16px;
    min-height: 420px;
    max-height: 520px;
    overflow-y: auto;
    margin-bottom: 12px;
    scroll-behavior: smooth;
}

/* ── Message bubbles ────────────────────────────────────────────────── */
.msg-row { display: flex; margin-bottom: 18px; align-items: flex-start; gap: 10px; }
.msg-row.user  { flex-direction: row-reverse; }
.msg-avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.msg-avatar.ai   { background: linear-gradient(135deg, #1e3a5f, #2563eb); }
.msg-avatar.user { background: linear-gradient(135deg, #1a3a2a, #22c55e); }
.msg-bubble {
    max-width: 82%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 0.88rem;
    line-height: 1.65;
    word-break: break-word;
}
.msg-bubble.ai {
    background: linear-gradient(135deg, #1a2540 0%, #1e2d47 100%);
    border: 1px solid #2a3a55;
    color: #cbd5e1;
    border-top-left-radius: 4px;
}
.msg-bubble.user {
    background: linear-gradient(135deg, #1a3a2a 0%, #166534 100%);
    border: 1px solid #1a4d2e;
    color: #d1fae5;
    border-top-right-radius: 4px;
}
.msg-time {
    font-size: 0.68rem;
    color: #4a5568;
    margin-top: 4px;
    text-align: right;
}

/* ── Typing indicator ───────────────────────────────────────────────── */
.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #2563eb;
    border-radius: 50%;
    margin: 0 2px;
    animation: typingBounce 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%, 80%, 100% { transform: translateY(0);    opacity: 0.4; }
    40%            { transform: translateY(-6px); opacity: 1.0; }
}

/* ── Chunk stats badge ──────────────────────────────────────────────── */
.chunk-badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.3);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #93b4f7;
    font-weight: 600;
    margin-top: 6px;
}
</style>
"""

st.set_page_config(
    page_title="LegalLens",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def get_model(ckpt_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, mode = load_checkpoint(ckpt_dir, device=device)
    return model, tokenizer, mode, device


PRESETS = {
    "Custom Input": {
        "lang": "en",
        "context": "",
        "original": "",
        "simplified": "",
    },
    "EN Contract - Obligation Shift (Shall -> May)": {
        "lang": "en",
        "context": "CONSULTING AGREEMENT - Section 4.3 (Payment Terms): Invoices and approval timelines.",
        "original": "Client shall pay all undisputed invoice amounts within thirty (30) days of receipt.",
        "simplified": "The client may choose to pay invoice amounts whenever convenient within 30 days.",
    },
    "EN Insurance - Exception Reversal": {
        "lang": "en",
        "context": "INSURANCE INDEMNITY - Section 8.4 (Exclusions): Specific hazards not covered by policy.",
        "original": "Insurer shall not be liable for any damage caused by flood, earthquake, or war, whether declared or undeclared.",
        "simplified": "The insurance company will cover damage resulting from floods and earthquakes during wartime.",
    },
    "EN Contract - Preserved Confidentiality": {
        "lang": "en",
        "context": "MASTER SERVICES AGREEMENT - Section 11.2 (Confidentiality Obligations)",
        "original": "The Receiving Party shall hold and maintain the Confidential Information in strictest confidence for the sole and exclusive benefit of the Disclosing Party.",
        "simplified": "The Receiving Party must keep the Confidential Information completely private and use it only for the benefit of the Disclosing Party.",
    },
    "HI Statute - Preserved Consumer Definition": {
        "lang": "hi",
        "context": "उपभोक्ता संरक्षण अधिनियम, 2019 - धारा 2(7): उपभोक्ता की कानूनी परिभाषा।",
        "original": "उपभोक्ता से ऐसा कोई व्यक्ति अभिप्रेत है जो किसी ऐसे प्रतिफल के लिए माल खरीदता है, जिसका संदाय किया जा चुका है या संदाय करने का वचन दिया गया है।",
        "simplified": "उपभोक्ता वह व्यक्ति है जो पैसे देकर या बाद में पैसे देने का वादा करके कोई सामान खरीदता है।",
    },
    "HI Contract - Liability Contradiction": {
        "lang": "hi",
        "context": "दायित्व परिसीमन खंड - खंड 11.2 (अधिकतम दायित्व सीमा)",
        "original": "इस अनुबंध के तहत विक्रेता का अधिकतम कानूनी दायित्व किसी भी परिस्थिति में ग्राहक द्वारा दी गई कुल फीस से अधिक नहीं होगा।",
        "simplified": "विक्रेता ग्राहक के किसी भी प्रकार के असीमित नुकसान की पूरी जिम्मेदारी उठाएगा।",
    },
}


def find_checkpoint():
    candidates = [
        os.path.join("checkpoints", "context_aware"),
        os.path.join("checkpoints", "sentence_only"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "context_aware"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "sentence_only"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "model.pt")):
            return c
    return None


def on_preset_change():
    """Callback when user selects a preset to cleanly update text area state."""
    selected = st.session_state.preset_select
    p = PRESETS[selected]
    st.session_state.context_val = p["context"]
    st.session_state.orig_val = p["original"]
    st.session_state.simp_val = p["simplified"]
    _lang_code_to_display = {"en": "English", "hi": "\u0939\u093f\u0928\u094d\u0926\u0940"}
    st.session_state.lang_select = _lang_code_to_display.get(p["lang"], "English")
    st.session_state.eval_mode_select = "context_aware" if p["context"].strip() else "sentence_only"
    st.session_state.finder_matches = []
    st.session_state.finder_needs_confirmation = False


def on_document_upload_change():
    """Callback triggered automatically when user uploads or changes a document image."""
    uploaded_file = st.session_state.get("doc_image_uploader")
    if uploaded_file is not None:
        lang_display = st.session_state.get("lang_select", "English")
        _lang_map = {"English": "en", "\u0939\u093f\u0928\u094d\u0926\u0940": "hi"}
        lang_code = _lang_map.get(lang_display, "en")
        ok, ocr_result, msg = extract_text_from_document_image(uploaded_file, lang=lang_code)
        if ok and ocr_result.strip():
            st.session_state.context_val = ocr_result.strip()
            st.session_state.eval_mode_select = "context_aware"
            words = len(ocr_result.split())
            st.session_state.ocr_msg_type = "toast"
            st.session_state.ocr_msg_text = f"Document OCR extracted {words} words into Context! Evaluation mode set to Context-Aware."
        else:
            st.session_state.ocr_msg_type = "error"
            st.session_state.ocr_msg_text = msg if msg else "No readable text detected in the uploaded image."


def generate_ai_simplified_clause_callback():
    """Callback to generate simplified clause using AI before widgets are instantiated."""
    orig_text = st.session_state.get("orig_val", "").strip()
    if not orig_text:
        st.session_state.ai_msg_type = "warning"
        st.session_state.ai_msg_text = "Please enter an Original Legal Clause first before generating a simplified version."
        return

    lang_display = st.session_state.get("lang_select", "English")
    _lang_map = {"English": "en", "\u0939\u093f\u0928\u094d\u0926\u0940": "hi"}
    lang_code = _lang_map.get(lang_display, "en")
    context = st.session_state.get("context_val", "")

    success, generated_simp, engine_name = generate_simplified_clause(
        original_text=orig_text,
        lang=lang_code,
        context=context,
    )
    if success and generated_simp:
        st.session_state.simp_val = generated_simp
        st.session_state.ai_msg_type = "toast"
        st.session_state.ai_msg_text = f"Simplified clause generated using {engine_name}!"
    else:
        st.session_state.ai_msg_type = "error"
        st.session_state.ai_msg_text = f"Generation error: {engine_name}"


def extract_ocr_callback():
    """Callback to extract OCR text from uploaded image before widgets are instantiated."""
    uploaded_file = st.session_state.get("doc_image_uploader")
    if uploaded_file is not None:
        lang_display = st.session_state.get("lang_select", "English")
        _lang_map = {"English": "en", "\u0939\u093f\u0928\u094d\u0926\u0940": "hi"}
        lang_code = _lang_map.get(lang_display, "en")
        ok, ocr_result, msg = extract_text_from_document_image(uploaded_file, lang=lang_code)
        if ok and ocr_result.strip():
            st.session_state.context_val = ocr_result.strip()
            st.session_state.eval_mode_select = "context_aware"
            words = len(ocr_result.split())
            st.session_state.ocr_msg_type = "toast"
            st.session_state.ocr_msg_text = f"Document OCR extracted {words} words into Context! Evaluation mode set to Context-Aware."
        else:
            st.session_state.ocr_msg_type = "error"
            st.session_state.ocr_msg_text = msg if msg else "No readable text detected in the uploaded image."
    else:
        st.session_state.ocr_msg_type = "warning"
        st.session_state.ocr_msg_text = "Please select or upload an image file first."


def clear_context_callback():
    """Callback to clear context before widgets are instantiated."""
    st.session_state.context_val = ""
    st.session_state.eval_mode_select = "sentence_only"
    st.session_state.finder_matches = []
    st.session_state.finder_needs_confirmation = False


def extract_clause_callback():
    """Callback for Smart Clause Finder search."""
    q = st.session_state.get("clause_search_query", "").strip()
    doc_text = st.session_state.get("context_val", "").strip()

    if not doc_text:
        st.session_state.finder_msg_type = "warning"
        st.session_state.finder_msg_text = "Please upload a legal document image and extract its text using OCR first."
        st.session_state.finder_matches = []
        st.session_state.finder_needs_confirmation = False
        return

    if not q:
        st.session_state.finder_msg_type = "warning"
        st.session_state.finder_msg_text = "Please enter a paragraph, point, clause, section, article number, or law name."
        st.session_state.finder_matches = []
        st.session_state.finder_needs_confirmation = False
        return

    matches = find_matching_clauses(doc_text, q)
    if not matches:
        st.session_state.finder_msg_type = "warning"
        st.session_state.finder_msg_text = f"No clause matching '{q}' was found in Document Context. Please verify the heading or section reference (e.g. '8. LIABILITY', 'Section 3', 'Point 3')."
        st.session_state.finder_matches = []
        st.session_state.finder_needs_confirmation = False
    elif len(matches) == 1:
        existing_orig = st.session_state.get("orig_val", "").strip()
        default_preset_texts = [p["original"].strip() for p in PRESETS.values() if p["original"].strip()]

        if not existing_orig or existing_orig in default_preset_texts:
            st.session_state.orig_val = matches[0]["text"]
            st.session_state.finder_msg_type = "success"
            st.session_state.finder_msg_text = f"Clause '{matches[0]['title']}' extracted and loaded successfully!"
            st.session_state.finder_matches = []
            st.session_state.finder_needs_confirmation = False
        else:
            # Require confirmation so user manual text is not overwritten unexpectedly
            st.session_state.finder_matches = matches
            st.session_state.finder_needs_confirmation = True
    else:
        # Multiple matches
        st.session_state.finder_matches = matches
        st.session_state.finder_needs_confirmation = False


def apply_selected_clause_callback(clause_text: str, clause_title: str):
    """Callback to apply a chosen clause into the Original Legal Clause area."""
    st.session_state.orig_val = clause_text
    st.session_state.finder_msg_type = "success"
    st.session_state.finder_msg_text = f"Clause '{clause_title}' applied successfully to Original Legal Clause!"
    st.session_state.finder_matches = []
    st.session_state.finder_needs_confirmation = False


def dismiss_clause_matches_callback():
    """Dismiss search results without overwriting."""
    st.session_state.finder_matches = []
    st.session_state.finder_needs_confirmation = False


# ---------------------------------------------------------------------------
# AI Document Chat Page
# ---------------------------------------------------------------------------

def _init_chat_state():
    """Initialize all chat-related session state keys.

    Keys are NEVER reset here — this is safe to call on every rerun,
    including when the user switches between sidebar tabs.
    """
    defaults = {
        "chat_history":           [],
        "chat_doc_text":          "",
        "chat_doc_name":          "",
        "chat_doc_size_str":      "",    # persisted so the card survives tab switches
        "chat_doc_chunks":        [],
        "chat_doc_index":         None,
        "chat_last_uploader_name": "",
        "chat_confirm_remove":    False,  # two-step remove confirmation
        "chat_pending_query":     "",     # example-question direct-submit
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _process_uploaded_doc(uploaded_file):
    """Extract text, chunk it, build vector index, and persist everything to session state."""
    with st.spinner("📄 Extracting document text..."):
        ok, text, msg = extract_text_from_document(uploaded_file, uploaded_file.name)

    if not ok:
        st.session_state.chat_doc_text = ""
        st.session_state.chat_doc_name = ""
        st.session_state.chat_doc_size_str = ""
        st.session_state.chat_doc_chunks = []
        st.session_state.chat_doc_index = None
        st.error(f"❌ {msg}")
        return False

    with st.spinner("🔍 Building search index..."):
        chunks = chunk_text(text)
        index = build_vector_index(chunks)

    # Persist size now so it survives tab navigation (file_uploader returns None on nav away)
    st.session_state.chat_doc_text = text
    st.session_state.chat_doc_name = uploaded_file.name
    st.session_state.chat_doc_size_str = format_file_size(uploaded_file)
    st.session_state.chat_doc_chunks = chunks
    st.session_state.chat_doc_index = index
    st.session_state.chat_history = []          # fresh chat for new document
    st.session_state.chat_last_uploader_name = uploaded_file.name
    st.session_state.chat_confirm_remove = False
    return True


def _clear_doc_state():
    """Wipe all document-related state without touching any evaluator state."""
    st.session_state.update({
        "chat_doc_text":          "",
        "chat_doc_name":          "",
        "chat_doc_size_str":      "",
        "chat_doc_chunks":        [],
        "chat_doc_index":         None,
        "chat_history":           [],
        "chat_last_uploader_name": "",
        "chat_confirm_remove":    False,
        "chat_pending_query":     "",
    })


def _handle_remove_click():
    """First click arms the confirmation; second (via Confirm btn) calls _clear_doc_state."""
    st.session_state.chat_confirm_remove = True


def _cancel_remove():
    st.session_state.chat_confirm_remove = False


def _send_query(query: str):
    """Append user message, generate AI answer, append it — all before next rerun."""
    if not query.strip():
        return
    if not st.session_state.get("chat_doc_text"):
        return
    # Guard against double-fire
    history = st.session_state.chat_history
    if history and history[-1]["role"] == "user" and history[-1]["content"] == query:
        return
    st.session_state.chat_history.append({"role": "user", "content": query})
    answer, _ = chat_with_document(
        query=query,
        chunks=st.session_state.chat_doc_chunks,
        index=st.session_state.chat_doc_index,
        history=st.session_state.chat_history[:-1],
        doc_name=st.session_state.chat_doc_name,
    )
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def render_chat_page():
    """Render the AI Document Chat page."""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    _init_chat_state()

    # ── Page header ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; margin-top:-15px; margin-bottom:22px;">
          <h1 style="color:#FFFFFF; font-size:2.4rem; font-weight:800;
                     letter-spacing:-0.5px; margin:0;">
            💬 &nbsp; AI Document Chat
          </h1>
          <p style="color:#6b82a8; font-size:0.95rem; margin-top:6px;">
            Upload a legal document and ask anything about it.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two-column layout ───────────────────────────────────────────────
    col_doc, col_chat = st.columns([1, 2.2], gap="large")

    # ════════════════════════════════════════════════════════════════════
    # LEFT PANEL – Document upload & info
    # ════════════════════════════════════════════════════════════════════
    with col_doc:
        st.markdown("### 📁 Document")

        # doc already in session state (e.g. user switched to Legal Evaluator and came back)
        persistent_doc_loaded = bool(st.session_state.get("chat_doc_text", ""))

        # Only show the uploader widget when NO document is currently loaded.
        # This keeps the widget key stable and avoids accidental clears on reruns.
        if not persistent_doc_loaded:
            uploaded_file = st.file_uploader(
                "Upload PDF or DOCX",
                type=["pdf", "docx"],
                key="chat_doc_uploader",
                help="Max 20 MB. Drag & drop or click Browse.",
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                is_valid, err_msg = validate_uploaded_file(uploaded_file, uploaded_file.name)
                if not is_valid:
                    st.error(f"❌ {err_msg}")
                else:
                    # Only process when it is a genuinely new file
                    if uploaded_file.name != st.session_state.chat_last_uploader_name:
                        _process_uploaded_doc(uploaded_file)
                        st.rerun()   # rerun so the uploader disappears and doc card appears
            else:
                # Empty state – no document loaded yet
                st.markdown(
                    """
                    <div class="empty-state">
                      <div class="empty-state-icon">📂</div>
                      <div class="empty-state-title">No document loaded</div>
                      <div class="empty-state-sub">
                        Drag &amp; drop a PDF or DOCX here,<br>or click <b>Browse files</b> above.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # ── Persistent document info card (shown even after tab switch) ──
            doc_name   = st.session_state.chat_doc_name
            size_str   = st.session_state.chat_doc_size_str or "—"
            chunk_count = len(st.session_state.chat_doc_chunks)
            word_count  = len(st.session_state.chat_doc_text.split())
            ext = doc_name.rsplit(".", 1)[-1].upper() if "." in doc_name else "DOC"

            st.markdown(
                f"""
                <div class="doc-card">
                  <div class="doc-card-title">Loaded Document</div>
                  <div class="doc-name">📄 {escape(doc_name)}</div>
                  <div class="doc-meta">Type: {escape(ext)} &nbsp;|&nbsp; Size: {escape(size_str)}</div>
                  <div class="doc-meta">Words: {word_count:,} &nbsp;|&nbsp; Chunks: {chunk_count}</div>
                  <div style="margin-top:8px;"><span class="doc-status-ok">✓ Ready to chat</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Quick preview
            with st.expander("🔍 Document Preview", expanded=False):
                preview_text = st.session_state.chat_doc_text[:1200]
                if len(st.session_state.chat_doc_text) > 1200:
                    preview_text += "\n\n*[... document continues ...]*"
                st.markdown(
                    f"<div style='font-size:0.8rem; color:#94a3b8; "
                    f"font-family:monospace; white-space:pre-wrap; "
                    f"max-height:260px; overflow-y:auto;'>{preview_text}</div>",
                    unsafe_allow_html=True,
                )

            # ── Two-step Remove Document ──────────────────────────────
            if st.session_state.get("chat_confirm_remove", False):
                st.warning(
                    "⚠️ **This will remove the document and clear all chat history.** "
                    "This action cannot be undone."
                )
                c_yes, c_no = st.columns(2)
                with c_yes:
                    st.button(
                        "✅ Confirm Remove",
                        key="chat_confirm_remove_yes",
                        type="primary",
                        use_container_width=True,
                        on_click=_clear_doc_state,
                    )
                with c_no:
                    st.button(
                        "Cancel",
                        key="chat_confirm_remove_no",
                        use_container_width=True,
                        on_click=_cancel_remove,
                    )
            else:
                st.button(
                    "🗑️ Remove Document",
                    key="chat_remove_doc",
                    use_container_width=True,
                    on_click=_handle_remove_click,
                )

        # ── Example questions ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<p style='color:#6b82a8; font-size:0.78rem; font-weight:700; "
            "text-transform:uppercase; letter-spacing:0.06em;'>Example questions</p>",
            unsafe_allow_html=True,
        )
        example_qs = [
            "What is this agreement about?",
            "Who are the parties involved?",
            "What is the duration of this contract?",
            "Explain the termination clause.",
            "What are the penalties for breach?",
            "Are there any risky clauses?",
        ]
        doc_ready = bool(st.session_state.get("chat_doc_text", ""))
        for q in example_qs:
            btn_clicked = st.button(
                q,
                key=f"eq_{abs(hash(q))}",
                use_container_width=True,
                disabled=not doc_ready,
            )
            if btn_clicked and doc_ready:
                # Directly process — no prefill, no extra click needed
                st.session_state.chat_pending_query = q
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # RIGHT PANEL – Chat interface
    # ════════════════════════════════════════════════════════════════════
    with col_chat:
        st.markdown("### 🤖 Chat")

        doc_loaded = bool(st.session_state.get("chat_doc_text", ""))
        history = st.session_state.get("chat_history", [])

        # ── Process a pending example-question query (set by left-panel button click) ──
        pending_q = st.session_state.pop("chat_pending_query", "")
        if pending_q and doc_loaded:
            with st.spinner("⚖️ Analysing document and generating response..."):
                _send_query(pending_q)
            # Refresh history after generation
            history = st.session_state.get("chat_history", [])

        # ── Chat message display ──────────────────────────────────────
        if not doc_loaded and not history:
            st.markdown(
                """
                <div class="chat-container">
                  <div class="empty-state" style="padding:50px 20px;">
                    <div class="empty-state-icon">💬</div>
                    <div class="empty-state-title">Start a conversation</div>
                    <div class="empty-state-sub">
                      Hello! Upload a legal document and ask me anything about it.<br><br>
                      I'll answer based on what's in your document and cite relevant sections.
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Render conversation history
            chat_html_parts = []

            if not history and doc_loaded:
                # First-time welcome after doc load
                doc_name = st.session_state.get("chat_doc_name", "your document")
                word_count = len(st.session_state.chat_doc_text.split())
                welcome_msg = (
                    f"\u2705 **Document loaded successfully!** I've indexed **{doc_name}** "
                    f"({word_count:,} words).\n\n"
                    f"You can now ask me anything about this document. I'll answer based "
                    f"only on what's written in it.\n\n"
                    f"\u26a0\ufe0f *This response is AI-generated and is not a substitute for "
                    f"professional legal advice.*"
                )
                chat_html_parts.append(
                    f'<div class="msg-row">'
                    f'<div class="msg-avatar ai">\u2696\ufe0f</div>'
                    f'<div>'
                    f'<div class="msg-bubble ai">{_md_to_html(welcome_msg)}</div>'
                    f'</div></div>'
                )

            for msg in history:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    chat_html_parts.append(
                        f'<div class="msg-row user">'
                        f'<div class="msg-avatar user">\U0001f464</div>'
                        f'<div>'
                        f'<div class="msg-bubble user">{_escape_html(content)}</div>'
                        f'</div></div>'
                    )
                else:
                    chat_html_parts.append(
                        f'<div class="msg-row">'
                        f'<div class="msg-avatar ai">\u2696\ufe0f</div>'
                        f'<div>'
                        f'<div class="msg-bubble ai">{_md_to_html(content)}</div>'
                        f'</div></div>'
                    )

            chat_html = "\n".join(chat_html_parts)
            st.markdown(
                f'<div class="chat-container" id="chat-scroll">{chat_html}</div>',
                unsafe_allow_html=True,
            )
            # Auto-scroll to bottom
            st.markdown(
                "<script>document.getElementById('chat-scroll').scrollTop = "
                "document.getElementById('chat-scroll').scrollHeight;</script>",
                unsafe_allow_html=True,
            )

        # \u2500\u2500 Top action bar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        bar_col1, bar_col2 = st.columns([5, 1])
        with bar_col2:
            if history:
                st.button(
                    "\U0001f5d1\ufe0f Clear",
                    key="chat_clear_btn",
                    use_container_width=True,
                    on_click=lambda: st.session_state.update({"chat_history": []}),
                    help="Clear conversation history",
                )

        # \u2500\u2500 Message input (manual typing) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        input_placeholder = (
            "Ask a question about your document..."
            if doc_loaded
            else "Upload a document to start chatting..."
        )

        with st.form(key="chat_input_form", clear_on_submit=True):
            input_col, btn_col = st.columns([6, 1])
            with input_col:
                user_input = st.text_input(
                    "Message",
                    placeholder=input_placeholder,
                    key="chat_message_input",
                    label_visibility="collapsed",
                    disabled=not doc_loaded,
                )
            with btn_col:
                send_btn = st.form_submit_button(
                    "\u27a4",
                    use_container_width=True,
                    type="primary",
                    disabled=not doc_loaded,
                )

        # \u2500\u2500 Handle manual send \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if send_btn and user_input and user_input.strip() and doc_loaded:
            with st.spinner("\u2696\ufe0f Analysing document and generating response..."):
                _send_query(user_input.strip())
            st.rerun()

        # \u2500\u2500 Status bar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if doc_loaded:
            chunk_count = len(st.session_state.chat_doc_chunks)
            msg_count = len([m for m in history if m["role"] == "user"])
            st.markdown(
                f"<p style='color:#4a5568; font-size:0.72rem; text-align:right; margin-top:4px;'>"
                f"\U0001f4da {chunk_count} indexed chunks &nbsp;|&nbsp; "
                f"\U0001f4ac {msg_count} question{'s' if msg_count != 1 else ''} asked"
                f"</p>",
                unsafe_allow_html=True,
            )

def main():
    # ── Sidebar navigation ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#e2e8f0; font-size:1.4rem; font-weight:800; "
            "margin-bottom:4px;'>⚖️ LegalLens</h2>"
            "<p style='color:#6b82a8; font-size:0.78rem; margin-top:0; "
            "margin-bottom:20px;'>AI-Powered Legal Analysis</p>",
            unsafe_allow_html=True,
        )
        nav_choice = st.radio(
            "Navigation",
            options=["⚖️  Legal Evaluator", "💬  AI Document Chat"],
            label_visibility="collapsed",
            key="nav_radio",
        )
        st.markdown("---")

    # ── Route to correct page ───────────────────────────────────────────
    if nav_choice == "💬  AI Document Chat":
        render_chat_page()
        return

    # ── Legal Evaluator page header ─────────────────────────────────────
    st.markdown(
        """
        <div style="text-align: center; margin-top: -15px; margin-bottom: 25px;">
            <h1 style="color: #FFFFFF; font-size: 2.75rem; font-weight: 800; letter-spacing: -0.5px; margin: 0; padding: 0;">
                ⚖️ &nbsp; LegalLens &nbsp; ⚖️
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state values if not present
    default_preset_key = list(PRESETS.keys())[1]  # "EN Contract - Obligation Shift (Shall -> May)"
    if "context_val" not in st.session_state:
        st.session_state.context_val = PRESETS[default_preset_key]["context"]
    if "orig_val" not in st.session_state:
        st.session_state.orig_val = PRESETS[default_preset_key]["original"]
    if "simp_val" not in st.session_state:
        st.session_state.simp_val = PRESETS[default_preset_key]["simplified"]
    if "lang_select" not in st.session_state:
        st.session_state.lang_select = "English"
    if "eval_mode_select" not in st.session_state:
        st.session_state.eval_mode_select = "context_aware" if st.session_state.context_val.strip() else "sentence_only"
    if "finder_matches" not in st.session_state:
        st.session_state.finder_matches = []
    if "finder_needs_confirmation" not in st.session_state:
        st.session_state.finder_needs_confirmation = False

    ckpt_dir = find_checkpoint()
    if not ckpt_dir:
        st.error(
            "Trained model checkpoint not found in `checkpoints/`. "
            "Please run `python article_src/src/run_all_experiments.py` first."
        )
        return

    try:
        with st.spinner("⚖️ Loading LegalLens evaluation engine and model weights..."):
            model, tokenizer, default_mode, device = get_model(ckpt_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not load the evaluation model: {exc}")
        st.info("Run git lfs pull and ensure the encoder is available locally or through Hugging Face.")
        return

    st.warning("This model is unvalidated and can miss changes to legal obligations. Review every result against the original text.")
    if default_mode == "sentence_only":
        st.caption("The loaded checkpoint was saved in sentence-only mode; context-aware performance is unverified.")

    tab1, tab2 = st.tabs(["🔍 Live Legal Evaluator", "📊 Experiments & Benchmark Results"])

    with tab1:
        col_left, col_right = st.columns([1.15, 1.0], gap="large")

        with col_left:
            st.subheader("Input Legal Clauses")
            preset_name = st.selectbox(
                "Load Sample Preset",
                list(PRESETS.keys()),
                index=1,
                key="preset_select",
                on_change=on_preset_change,
            )

            c1, c2 = st.columns(2)
            with c1:
                lang_display = st.selectbox(
                    "Language",
                    options=["English", "\u0939\u093f\u0928\u094d\u0926\u0940"],
                    key="lang_select",
                )
                _lang_map = {"English": "en", "\u0939\u093f\u0928\u094d\u0926\u0940": "hi"}
                lang = _lang_map.get(lang_display, "en")
            with c2:
                eval_mode = st.selectbox(
                    "Evaluation Mode",
                    options=["context_aware", "sentence_only"],
                    key="eval_mode_select",
                    help="Context-Aware evaluates the clause within surrounding document/statute context. Sentence-Only evaluates the standalone pair.",
                )

            # Document Image Upload with OCR
            with st.expander("📷 Upload Legal Document Image (OCR Extraction)", expanded=True):
                uploaded_doc = st.file_uploader(
                    "Choose an image of a legal document or section (PNG, JPG, JPEG)",
                    type=["png", "jpg", "jpeg"],
                    key="doc_image_uploader",
                    on_change=on_document_upload_change,
                )
                if uploaded_doc is not None:
                    st.image(uploaded_doc, caption=f"Document Preview: {uploaded_doc.name}", use_container_width=True)
                    c_ocr1, c_ocr2 = st.columns([1.7, 1])
                    with c_ocr1:
                        st.button(
                            "📄 Extract Text with OCR into Context",
                            key="ocr_extract_btn",
                            type="secondary",
                            use_container_width=True,
                            on_click=extract_ocr_callback,
                        )
                    with c_ocr2:
                        st.button(
                            "🗑️ Clear Context",
                            key="clear_ctx_btn",
                            use_container_width=True,
                            on_click=clear_context_callback,
                        )

            # Document / Section Context Area (Populated by OCR or manual input, fully editable)
            st.text_area(
                "Document / Section Context",
                key="context_val",
                height=110,
                help="Enter contract headings, statute chapters, surrounding paragraph context, or text extracted automatically via OCR. You can edit this freely.",
            )

            ocr_msg_type = st.session_state.pop("ocr_msg_type", None)
            ocr_msg_text = st.session_state.pop("ocr_msg_text", None)
            if ocr_msg_type == "toast":
                st.toast(f"✅ {ocr_msg_text}", icon="📄")
            elif ocr_msg_type == "error":
                st.error(ocr_msg_text)

            # -----------------------------------------------------------------
            # Smart Clause Finder Section
            # -----------------------------------------------------------------
            st.subheader("Smart Clause Finder")

            c_fin1, c_fin2 = st.columns([2.3, 1], gap="small",vertical_alignment="bottom")
            with c_fin1:
                st.text_input(
                    "Enter Paragraph / Point / Clause / Section / Article Number or Law Name",
                    placeholder="e.g. Clause 4.3, Section 8, Article 12, Indian Contract Act...",
                    key="clause_search_query",
                    help="Locates specific clauses or statutory provisions directly from the document context.",
                )
            with c_fin2:
                st.write("")
                st.write("")
                st.button(
                    "Extract Clause",
                    key="extract_clause_btn",
                    type="secondary",
                    use_container_width=True,
                    on_click=extract_clause_callback,
                )

            # Render feedback messages from clause extraction
            finder_msg_type = st.session_state.pop("finder_msg_type", None)
            finder_msg_text = st.session_state.pop("finder_msg_text", None)
            if finder_msg_type == "success":
                st.success(f"✅ {finder_msg_text}")
            elif finder_msg_type == "warning":
                st.warning(finder_msg_text)
            elif finder_msg_type == "info":
                st.info(finder_msg_text)
            elif finder_msg_type == "error":
                st.error(finder_msg_text)

            # Handle multiple results or manual overwrite confirmation
            matches = st.session_state.get("finder_matches", [])
            needs_confirm = st.session_state.get("finder_needs_confirmation", False)

            if matches:
                if needs_confirm and len(matches) == 1:
                    m = matches[0]
                    st.info(f"**Found Clause:** `{m['title']}`\n\n_{m['preview']}_")
                    st.warning("⚠️ Your Original Legal Clause already contains manual text. Do you want to replace it?")
                    c_conf1, c_conf2 = st.columns([1.5, 1])
                    with c_conf1:
                        st.button(
                            "✅ Confirm & Overwrite Clause",
                            key="confirm_overwrite_btn",
                            type="primary",
                            use_container_width=True,
                            on_click=apply_selected_clause_callback,
                            args=(m["text"], m["title"]),
                        )
                    with c_conf2:
                        st.button(
                            "Cancel",
                            key="cancel_overwrite_btn",
                            use_container_width=True,
                            on_click=dismiss_clause_matches_callback,
                        )
                elif len(matches) > 1:
                    st.info(f"**Found {len(matches)} matching clauses.** Please select which one to populate:")
                    match_labels = [f"{i+1}. {m['title']}" for i, m in enumerate(matches)]
                    selected_label = st.selectbox(
                        "Matching Clauses",
                        options=match_labels,
                        key="multi_match_select",
                    )
                    selected_idx = match_labels.index(selected_label)
                    chosen_m = matches[selected_idx]
                    st.caption(f"**Preview:** {chosen_m['preview']}")

                    c_sel1, c_sel2 = st.columns([1.5, 1])
                    with c_sel1:
                        st.button(
                            "📋 Populate Selected Clause",
                            key="apply_multi_btn",
                            type="primary",
                            use_container_width=True,
                            on_click=apply_selected_clause_callback,
                            args=(chosen_m["text"], chosen_m["title"]),
                        )
                    with c_sel2:
                        st.button(
                            "Dismiss",
                            key="dismiss_multi_btn",
                            use_container_width=True,
                            on_click=dismiss_clause_matches_callback,
                        )

            # Original Legal Clause Area
            st.text_area(
                "Original Legal Clause",
                key="orig_val",
                height=110,
                help="The formal legal clause or statutory provision to be evaluated.",
            )

            # Simplified Candidate Clause Area
            st.text_area(
                "Simplified Candidate Clause",
                key="simp_val",
                height=110,
                help="The plain-language simplified candidate clause. You can generate this using AI below or enter it manually.",
            )

            # AI-Based Simplification Button (using on_click callback to prevent StreamlitWidgetAlreadyInstantiatedError)
            st.button(
                "✨ Generate Simplified Version with AI",
                key="ai_generate_btn",
                type="secondary",
                use_container_width=True,
                on_click=generate_ai_simplified_clause_callback,
            )

            # Display notifications from AI generation callback if present
            ai_msg_type = st.session_state.pop("ai_msg_type", None)
            ai_msg_text = st.session_state.pop("ai_msg_text", None)
            if ai_msg_type == "toast":
                st.toast(f"✅ {ai_msg_text}", icon="✨")
            elif ai_msg_type == "warning":
                st.warning(ai_msg_text)
            elif ai_msg_type == "error":
                st.error(ai_msg_text)

            st.markdown("---")
            analyze_btn = st.button("⚖️ Evaluate Legal Meaning Preservation", type="primary", use_container_width=True)

        with col_right:
            st.subheader("LegalLens Assessment")
            context_text = st.session_state.context_val
            original_text = st.session_state.orig_val
            simplified_text = st.session_state.simp_val
            active_mode = st.session_state.get("eval_mode_select", "context_aware" if context_text.strip() else "sentence_only")

            if analyze_btn or preset_name != "Custom Input":
                if not original_text.strip() or not simplified_text.strip():
                    st.warning("Please enter both an Original Legal Clause and a Simplified Candidate Clause.")
                else:
                    effective_mode = active_mode if context_text.strip() else "sentence_only"
                    with st.spinner("⚖️ Evaluating semantic & legal meaning preservation..."):
                        pred = predict_single_pair(
                            model,
                            tokenizer,
                            original=original_text,
                            simplification=simplified_text,
                            context=context_text,
                            lang=lang,
                            mode=effective_mode,
                            device=device,
                        )

                    score = pred["lmp_score"]
                    decision = pred["decision"]
                    context_used = pred.get("context_used", False)

                    # Confirmation indicator for Document Context utilization
                    if context_used:
                        ctx_words = len(context_text.split())
                        st.success(
                            f"🌐 **Context-Aware Evaluation Active** — Clause evaluated within attached document context "
                            f"({ctx_words} words | {lang_display})."
                        )
                    else:
                        if context_text.strip() and active_mode == "sentence_only":
                            st.info(
                                f"📄 **Sentence-Only Evaluation Mode** (Document context bypassed per user mode selection | {lang_display})."
                            )
                        else:
                            st.info(f"📄 **Sentence-Only Evaluation Mode** (No document context attached | {lang_display}).")

                    m1, m2, m3 = st.columns([1.1, 1.1, 1.1])
                    m1.metric("LMP Score (1 – 10)", f"{score:.2f} / 10.0")
                    if decision == "Preserved":
                        m2.success(f"**Decision:** {decision}")
                    elif decision == "Partially Preserved":
                        m2.warning(f"**Decision:** {decision}")
                    else:
                        m2.error(f"**Decision:** {decision}")

                    mode_badge = "Context-Aware" if context_used else "Sentence-Only"
                    m3.metric("Eval Mode", mode_badge)

                    st.progress(min(1.0, max(0.0, (score - 1.0) / 9.0)))

                    errs = pred["error_types"]
                    err_str = ", ".join(errs) if errs != ["none"] else "None (Faithfully Preserved)"
                    st.markdown(f"**Detected Legal Error Types:** `{err_str}`")
                    st.info(f"**Explanation:** {pred['explanation']}")

                    with st.expander("Diagnostic Error Probabilities (9 Legal Dimensions)", expanded=True):
                        err_df = pd.DataFrame(
                            [
                                {"Error Dimension": k, "Probability": v}
                                for k, v in pred["error_probabilities"].items()
                            ]
                        ).sort_values("Probability", ascending=False)
                        st.bar_chart(err_df.set_index("Error Dimension"))

    with tab2:
        st.subheader("Empirical Benchmark Results & Evaluation Experiments")
        summary_path = os.path.join("predictions", "experiment_results_summary.json")
        cases_path = os.path.join("predictions", "experiment_5_error_case_studies.json")

        if not os.path.exists(summary_path):
            st.info("No measured experiment results are available. Run the experiment pipeline to generate them.")

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            exp1 = summary.get("experiment_1_baseline_vs_judgebert", {})
            exp2 = summary.get("experiment_2_cross_lingual_evaluation", summary.get("experiment_2_cross_lingual", {}))
            exp3 = summary.get("experiment_3_context_aware_generalization", {})
            exp4 = summary.get("experiment_4_sanity_checks", {})

            st.markdown("### Experiments 1–3: Baselines, Cross-Lingual & Context Comparison")
            rows = []
            if "generic_sbert_baseline" in exp1:
                b1 = exp1["generic_sbert_baseline"]
                rows.append({"Model / Split": b1.get("model", "Generic SBERT"), "Pearson r": b1.get("pearson_r", 0.0), "RMSE": b1.get("rmse", 0.0), "Decision Acc": "N/A"})
            if "lexical_tfidf_baseline" in exp1:
                b2 = exp1["lexical_tfidf_baseline"]
                rows.append({"Model / Split": b2.get("model", "TF-IDF"), "Pearson r": b2.get("pearson_r", 0.0), "RMSE": b2.get("rmse", 0.0), "Decision Acc": "N/A"})
            if "sentence_only" in exp3:
                s = exp3["sentence_only"]
                s_acc = s.get("decision_accuracy", s.get("decision_acc", s.get("acc", 0.0)))
                rows.append({"Model / Split": "LegalLens (Sentence-Only)", "Pearson r": s.get("pearson_r", 0.0), "RMSE": s.get("rmse", 0.0), "Decision Acc": f"{s_acc*100:.1f}%"})
            if "context_aware" in exp3:
                c = exp3["context_aware"]
                c_acc = c.get("decision_accuracy", c.get("decision_acc", c.get("acc", 0.0)))
                rows.append({"Model / Split": "LegalLens (Context-Aware)", "Pearson r": c.get("pearson_r", 0.0), "RMSE": c.get("rmse", 0.0), "Decision Acc": f"{c_acc*100:.1f}%"})
            
            e = exp2.get("english_subcorpus", exp2.get("english", {}))
            if e:
                e_acc = e.get("decision_accuracy", e.get("acc", 0.0))
                rows.append({"Model / Split": "Cross-Lingual: English Sub-Corpus", "Pearson r": e.get("pearson_r", 0.0), "RMSE": e.get("rmse", 0.0), "Decision Acc": f"{e_acc*100:.1f}%"})
            
            h = exp2.get("hindi_subcorpus", exp2.get("hindi", {}))
            if h:
                h_acc = h.get("decision_accuracy", h.get("acc", 0.0))
                rows.append({"Model / Split": "Cross-Lingual: Hindi Sub-Corpus", "Pearson r": h.get("pearson_r", 0.0), "RMSE": h.get("rmse", 0.0), "Decision Acc": f"{h_acc*100:.1f}%"})

            st.table(pd.DataFrame(rows))

            if exp4:
                st.markdown("### Experiment 4: Sanity Checks (Identical vs. Unrelated Pairs)")
                id_info = exp4.get("identical_pairs_sanity", {})
                un_info = exp4.get("unrelated_pairs_sanity", {})
                id_mean = id_info.get("mean_predicted_score", exp4.get("identical_mean_pred"))
                un_mean = un_info.get("mean_predicted_score", exp4.get("unrelated_mean_pred"))

                sc1, sc2 = st.columns(2)
                sc1.metric("LegalLens Identical Mean", f"{id_mean:.2f} / 10.0" if id_mean is not None else "Not measured", "Target: 10.0")
                sc2.metric("LegalLens Unrelated Mean", f"{un_mean:.2f} / 10.0" if un_mean is not None else "Not measured", "Target: 1.0 (Low)")
                for label, key in (
                    ("Generic SBERT Identical", "generic_sbert_identical_mean"),
                    ("Generic SBERT Unrelated", "generic_sbert_unrelated_mean"),
                ):
                    value = exp4.get(key)
                    if value is not None:
                        st.metric(label, f"{value:.2f} / 10.0")

        if os.path.exists(cases_path):
            with open(cases_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            st.markdown(f"### Qualitative Error Analysis ({len(cases)} Case Studies)")
            case_rows = []
            for c in cases:
                gt_score = c.get("ground_truth_score", c.get("human_ground_truth", {}).get("lmp_score", "-"))
                gt_dec = c.get("ground_truth_decision", c.get("human_ground_truth", {}).get("decision", "-"))
                jb_score = c.get("judgebert_predicted_score", c.get("judgebert_prediction", {}).get("predicted_lmp_score", "-"))
                jb_dec = c.get("judgebert_decision", c.get("judgebert_prediction", {}).get("predicted_decision", "-"))
                err_val = c.get("diagnosed_error")
                if not err_val:
                    err_val = ", ".join(c.get("judgebert_prediction", {}).get("diagnosed_error_types", ["none"]))
                
                case_rows.append({
                    "ID": c.get("pair_id", ""),
                    "Lang": c.get("language", "").upper(),
                    "Domain": c.get("domain", "").capitalize(),
                    "Dimension / Shift": c.get("dimension", ""),
                    "GT Score": gt_score,
                    "GT Decision": gt_dec,
                    "Generic SBERT": c.get("generic_sbert_score", "-"),
                    "LegalLens Score": jb_score,
                    "Pred Decision": jb_dec,
                    "Diagnosed Error": err_val,
                    "Explanation": c.get("explanation", ""),
                })
            st.dataframe(pd.DataFrame(case_rows), use_container_width=True)


if __name__ == "__main__":
    main()
