"""
🎙️ Voice-Controlled Local AI Agent
Mem0 AI/ML & Generative AI Developer Intern Assignment

A local AI agent that accepts audio input, classifies intent using Ollama,
executes tools (file ops, code gen, summarize, chat), and displays results
in a premium Streamlit interface.

Features:
- Dual audio input (microphone + file upload)
- Multi-language speech-to-text (faster-whisper)
- Compound command support (multiple intents per input)
- Human-in-the-loop confirmation for file operations
- Session memory with conversation context
- Graceful error degradation
- Model benchmarking dashboard
"""

import streamlit as st
import os
import time

# Agent modules
from agent.stt import SpeechToText, SUPPORTED_LANGUAGES
from agent.intent import IntentClassifier
from agent.tools import (
    create_file, create_folder, generate_code,
    summarize_text, general_chat, list_output_files, OUTPUT_DIR
)
from agent.memory import SessionMemory

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent — Mem0",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Premium Dark Glassmorphism Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    html, body, [class*="st-"], p, span, div, h1, h2, h3, h4, h5, h6, label, input, textarea, button, a, li, td, th {
        font-family: 'Inter', sans-serif !important;
    }

    /* Preserve Streamlit icon fonts */
    [data-testid="stIcon"], .material-icons, [class*="icon"] {
        font-family: inherit !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #161B22 50%, #0E1117 100%);
    }

    /* Fix file uploader styling to prevent overlap */
    [data-testid="stFileUploader"] {
        padding: 16px 0;
    }

    [data-testid="stFileUploader"] section {
        padding: 16px;
        border-radius: 12px;
        border: 1px dashed rgba(108, 99, 255, 0.3);
        background: rgba(22, 27, 34, 0.4);
    }

    [data-testid="stFileUploader"] section > button {
        margin-top: 8px;
    }

    /* Fix upload button icon rendering as duplicate text */
    [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] span[data-testid="stIconMaterial"] {
        display: none;
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4ECDC4 50%, #44CF6C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #8B949E;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(108, 99, 255, 0.35);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.1);
    }

    .glass-card-success {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(68, 207, 108, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .glass-card-warning {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 193, 7, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .glass-card-danger {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 82, 82, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #E6EDF3;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #6C63FF;
        margin-bottom: 6px;
    }

    /* ── Pipeline Stepper ── */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        margin-bottom: 20px;
    }

    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        flex: 1;
    }

    .step-circle {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: 700;
        transition: all 0.4s ease;
    }

    .step-inactive {
        background: rgba(139, 148, 158, 0.15);
        color: #8B949E;
        border: 2px solid rgba(139, 148, 158, 0.2);
    }

    .step-active {
        background: linear-gradient(135deg, #6C63FF, #4ECDC4);
        color: #fff;
        border: 2px solid #6C63FF;
        box-shadow: 0 0 20px rgba(108, 99, 255, 0.4);
        animation: pulse-glow 2s infinite;
    }

    .step-done {
        background: rgba(68, 207, 108, 0.2);
        color: #44CF6C;
        border: 2px solid rgba(68, 207, 108, 0.4);
    }

    .step-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .step-connector {
        flex: 0.5;
        height: 2px;
        background: rgba(139, 148, 158, 0.2);
        margin-top: -20px;
    }

    .step-connector-done {
        flex: 0.5;
        height: 2px;
        background: linear-gradient(90deg, #44CF6C, #4ECDC4);
        margin-top: -20px;
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(108, 99, 255, 0.4); }
        50% { box-shadow: 0 0 30px rgba(108, 99, 255, 0.7); }
    }

    /* ── Intent Badges ── */
    .intent-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px;
    }

    .badge-create-file {
        background: rgba(78, 205, 196, 0.15);
        color: #4ECDC4;
        border: 1px solid rgba(78, 205, 196, 0.3);
    }

    .badge-write-code {
        background: rgba(108, 99, 255, 0.15);
        color: #6C63FF;
        border: 1px solid rgba(108, 99, 255, 0.3);
    }

    .badge-summarize {
        background: rgba(255, 193, 7, 0.15);
        color: #FFC107;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }

    .badge-general-chat {
        background: rgba(232, 121, 249, 0.15);
        color: #E879F9;
        border: 1px solid rgba(232, 121, 249, 0.3);
    }

    /* ── Stats Metrics ── */
    .metric-card {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(108, 99, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Sidebar Styles ── */
    section[data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.95);
        border-right: 1px solid rgba(108, 99, 255, 0.1);
    }

    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        color: #E6EDF3;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(108, 99, 255, 0.2);
    }

    /* ── History Entry ── */
    .history-entry {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(108, 99, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }

    .history-entry:hover {
        border-color: rgba(108, 99, 255, 0.3);
    }

    .history-time {
        font-size: 0.65rem;
        color: #6C63FF;
        font-weight: 600;
    }

    .history-text {
        font-size: 0.8rem;
        color: #E6EDF3;
        margin-top: 4px;
        line-height: 1.4;
    }

    .history-status {
        font-size: 0.65rem;
        margin-top: 6px;
        font-weight: 600;
    }

    /* ── Benchmark Bar ── */
    .bench-bar {
        height: 6px;
        border-radius: 3px;
        margin: 4px 0 8px 0;
    }

    /* ── Confirmation Card ── */
    .confirm-card {
        background: rgba(22, 27, 34, 0.9);
        border: 2px solid rgba(255, 193, 7, 0.4);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }

    /* ── Status indicator ── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }

    .dot-green {
        background: #44CF6C;
        box-shadow: 0 0 8px rgba(68, 207, 108, 0.5);
    }

    .dot-red {
        background: #FF5252;
        box-shadow: 0 0 8px rgba(255, 82, 82, 0.5);
    }

    .dot-yellow {
        background: #FFC107;
        box-shadow: 0 0 8px rgba(255, 193, 7, 0.5);
    }

    /* ── File Tree ── */
    .file-tree-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        background: rgba(22, 27, 34, 0.4);
        border-radius: 8px;
        margin: 4px 0;
        font-size: 0.85rem;
        color: #E6EDF3;
    }

    /* ── Responsive tweaks ── */
    @media (max-width: 768px) {
        .hero-header { font-size: 1.8rem; }
        .pipeline-container { flex-wrap: wrap; gap: 8px; }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
if "stt_engine" not in st.session_state:
    st.session_state.stt_engine = SpeechToText(model_size="base")
if "intent_engine" not in st.session_state:
    st.session_state.intent_engine = IntentClassifier(model="mistral")
if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = 0
if "current_transcription" not in st.session_state:
    st.session_state.current_transcription = None
if "current_intents" not in st.session_state:
    st.session_state.current_intents = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "pending_confirmations" not in st.session_state:
    st.session_state.pending_confirmations = []
if "confirm_context" not in st.session_state:
    st.session_state.confirm_context = {}
if "benchmarks" not in st.session_state:
    st.session_state.benchmarks = {}
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "mistral"

# Ensure output and uploads directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def get_intent_badge(intent_type: str) -> str:
    """Return HTML for a styled intent badge."""
    icons = {
        "create_file": "📁", "write_code": "💻",
        "summarize": "📝", "general_chat": "💬",
    }
    icon = icons.get(intent_type, "⚡")
    css_class = f"badge-{intent_type.replace('_', '-')}"
    label = intent_type.replace("_", " ").title()
    return f'<span class="intent-badge {css_class}">{icon} {label}</span>'


def render_pipeline(step: int):
    """Render the pipeline progress stepper."""
    steps = [
        ("🎤", "Audio"),
        ("🗣️", "STT"),
        ("🧠", "Intent"),
        ("⚡", "Execute"),
        ("✅", "Done"),
    ]

    html = '<div class="pipeline-container">'
    for i, (icon, label) in enumerate(steps):
        if i < step:
            css = "step-done"
            display_icon = "✓"
        elif i == step:
            css = "step-active"
            display_icon = icon
        else:
            css = "step-inactive"
            display_icon = icon

        html += f'''
        <div class="pipeline-step">
            <div class="step-circle {css}">{display_icon}</div>
            <span class="step-label">{label}</span>
        </div>'''

        if i < len(steps) - 1:
            conn_class = "step-connector-done" if i < step else "step-connector"
            html += f'<div class="{conn_class}"></div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_benchmarks(benchmarks: dict):
    """Render benchmark results."""
    if not benchmarks:
        return

    total = sum(v for v in benchmarks.values() if isinstance(v, (int, float)))
    if total == 0:
        return

    cols = st.columns(len(benchmarks))
    colors = {
        "stt_time": "#4ECDC4",
        "intent_time": "#6C63FF",
        "tool_time": "#44CF6C",
    }
    labels = {
        "stt_time": "Speech-to-Text",
        "intent_time": "Intent Classification",
        "tool_time": "Tool Execution",
    }

    for i, (key, value) in enumerate(benchmarks.items()):
        if isinstance(value, (int, float)):
            color = colors.get(key, "#8B949E")
            label = labels.get(key, key)
            pct = round((value / total) * 100) if total > 0 else 0
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}s</div>
                    <div class="metric-label">{label}</div>
                    <div class="bench-bar" style="background: linear-gradient(90deg, {color} {pct}%, rgba(139,148,158,0.15) {pct}%);"></div>
                </div>
                """, unsafe_allow_html=True)


def execute_intent(intent: dict, transcription: str, model: str) -> dict:
    """Execute a single intent and return the result."""
    intent_type = intent.get("type", "general_chat")
    params = intent.get("params", {})

    if intent_type == "create_file":
        filename = params.get("filename", "untitled.txt")
        file_type = params.get("file_type", "file")
        if file_type == "folder":
            return create_folder(filename)
        else:
            content = params.get("content", "")
            return create_file(filename, content)

    elif intent_type == "write_code":
        desc = params.get("description", transcription)
        lang = params.get("language", "python")
        fname = params.get("filename", None)
        return generate_code(desc, lang, fname, model=model)

    elif intent_type == "summarize":
        text = params.get("text_to_summarize", transcription)
        return summarize_text(text, model=model)

    elif intent_type == "general_chat":
        message = params.get("message", transcription)
        history = st.session_state.memory.get_conversation_context()
        return general_chat(message, history, model=model)

    else:
        # Fallback to general chat
        history = st.session_state.memory.get_conversation_context()
        return general_chat(transcription, history, model=model)


def needs_confirmation(intent_type: str) -> bool:
    """Check if an intent requires human confirmation."""
    return intent_type in ("create_file", "write_code")


def enrich_intents_with_results(intents: list, previous_results: list):
    """
    Pipe output from completed intents into subsequent intents.
    Enables compound commands like 'Summarize and save to file' by
    injecting the summary/code as file content automatically.
    """
    if not previous_results:
        return

    # Collect produced content from previous results
    produced_content = None
    for r in previous_results:
        if not isinstance(r, dict) or not r.get("success"):
            continue
        if r.get("action") == "summarize" and r.get("summary"):
            produced_content = r["summary"]
        elif r.get("action") == "write_code" and r.get("code"):
            produced_content = r["code"]
        elif r.get("action") == "general_chat" and r.get("response"):
            produced_content = r["response"]

    if not produced_content:
        return

    # Inject content into file creation intents that lack content
    for intent in intents:
        if intent.get("type") == "create_file":
            params = intent.get("params", {})
            if not params.get("content"):
                params["content"] = produced_content
                intent["params"] = params


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎙️ Voice AI Agent</div>', unsafe_allow_html=True)

    st.divider()

    # Ollama connection status
    st.markdown('<div class="sidebar-title">📡 System Status</div>', unsafe_allow_html=True)
    ollama_connected = st.session_state.intent_engine.check_connection()
    if ollama_connected:
        st.markdown(
            '<span class="status-dot dot-green"></span> <span style="color:#44CF6C;font-size:0.85rem;font-weight:600;">Ollama Connected</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-dot dot-red"></span> <span style="color:#FF5252;font-size:0.85rem;font-weight:600;">Ollama Disconnected</span>',
            unsafe_allow_html=True
        )
        st.caption("Run `ollama serve` and `ollama pull mistral`")

    st.markdown(
        '<span class="status-dot dot-green"></span> <span style="color:#44CF6C;font-size:0.85rem;font-weight:600;">Whisper Ready</span>',
        unsafe_allow_html=True
    )

    st.divider()

    # Session Memory / History
    st.markdown('<div class="sidebar-title">📜 Session History</div>', unsafe_allow_html=True)
    history = st.session_state.memory.get_history()

    if history:
        for entry in reversed(history):
            status_color = "#44CF6C" if entry["status"] == "completed" else "#FF5252"
            intent_types = ", ".join([i.get("type", "?").replace("_", " ").title() for i in entry.get("intents", [])])
            if not intent_types:
                intent_types = "—"

            st.markdown(f"""
            <div class="history-entry">
                <div class="history-time">🕐 {entry['timestamp']}</div>
                <div class="history-text">"{entry['transcription'][:60]}{"..." if len(entry.get('transcription','')) > 60 else ''}"</div>
                <div class="history-status" style="color: {status_color};">
                    ● {entry['status'].title()} — {intent_types}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No commands yet. Record or upload audio to get started!")

    st.divider()

    # Session Stats
    stats = st.session_state.memory.get_total_stats()
    if stats["total_commands"] > 0:
        st.markdown('<div class="sidebar-title">📊 Session Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Commands", stats["total_commands"])
        c2.metric("Files", stats["files_created"] + stats["code_generated"])
        c1.metric("Summaries", stats["summaries"])
        c2.metric("Chats", stats["chats"])

        if stats["total_processing_time"] > 0:
            st.caption(f"⏱ Total processing: {stats['total_processing_time']}s")
            st.caption(f"Avg STT: {stats['avg_stt_time']}s | Avg Intent: {stats['avg_intent_time']}s")


# ─────────────────────────────────────────────
# Main Content Area
# ─────────────────────────────────────────────

# Hero Header
st.markdown('<div class="hero-header">🎙️ Voice AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Speak your command. I\'ll transcribe, understand, and execute — all locally.</div>', unsafe_allow_html=True)

# Pipeline Stepper
render_pipeline(st.session_state.pipeline_step)

# ── Model & Language Settings (always visible on main page) ──
st.markdown('<div class="section-label">CONFIGURATION</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">⚙️ Model & Language Settings</div>', unsafe_allow_html=True)

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

with cfg_col1:
    model = st.selectbox(
        "🤖 LLM Model (Ollama)",
        ["mistral", "llama3", "llama3.1", "codellama", "gemma", "phi3"],
        index=0,
        key="main_model_select",
        help="Select the Ollama model for intent classification and tool execution."
    )
    st.session_state.ollama_model = model

with cfg_col2:
    whisper_size = st.selectbox(
        "🗣️ Whisper Model Size",
        ["tiny", "base", "small", "medium"],
        index=1,
        key="main_whisper_select",
        help="Larger = more accurate but slower. 'base' is recommended."
    )
    if whisper_size != st.session_state.stt_engine.model_size:
        st.session_state.stt_engine = SpeechToText(model_size=whisper_size)

with cfg_col3:
    language = st.selectbox(
        "🌐 Audio Language",
        list(SUPPORTED_LANGUAGES.keys()),
        index=0,
        key="main_lang_select",
        help="Select the spoken language, or use Auto Detect."
    )
    selected_language = SUPPORTED_LANGUAGES[language]

# ── STEP 1: Input ──
st.markdown('<div class="section-label">STEP 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">🎤 Input — Voice or Text</div>', unsafe_allow_html=True)

audio_file_path = None
text_prompt_input = None

# All inputs visible permanently — no tabs
col_audio, col_upload = st.columns(2)

with col_audio:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-label">🎙️ RECORD MICROPHONE</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("🔄 Reset", use_container_width=True, key="clear_audio_btn"):
            st.session_state.pipeline_step = 0
            st.session_state.current_transcription = None
            st.session_state.current_intents = None
            st.session_state.current_results = None
            st.session_state.pending_confirmations = []
            st.session_state.confirm_context = {}
            st.session_state.benchmarks = {}
            st.session_state.audio_processed = False
            # Clear widgets
            for key in ["mic_input", "file_upload", "text_prompt"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            
    audio_value = st.audio_input("Click to start recording", key="mic_input")
    if audio_value:
        rec_path = os.path.join("uploads", "recorded_audio.wav")
        with open(rec_path, "wb") as f:
            f.write(audio_value.getbuffer())
        audio_file_path = rec_path
        st.success("✅ Audio recorded!")

with col_upload:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">📁 UPLOAD AUDIO FILE</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop an audio file here",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="file_upload",
        label_visibility="collapsed"
    )
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_file_path = file_path
        st.success(f"✅ Uploaded: `{uploaded_file.name}`")

# Audio playback
if audio_file_path:
    st.audio(audio_file_path)

# ── Text Prompt (always visible) ──
st.markdown("""
<div class="glass-card">
    <div class="section-label">✍️ YOUR PROMPT / INSTRUCTIONS</div>
    <p style="font-size:0.8rem;color:#8B949E;margin-top:4px;">
        Type your command, or add instructions to go along with the audio above.
    </p>
</div>
""", unsafe_allow_html=True)
text_prompt_input = st.text_area(
    "Your prompt or instructions",
    placeholder='e.g. "Summarize the above audio" or "Create a Python file with a retry function"',
    height=100,
    key="text_prompt",
    label_visibility="collapsed"
)
if not text_prompt_input or not text_prompt_input.strip():
    text_prompt_input = None

# Determine if we have any input to process
has_input = audio_file_path is not None or (text_prompt_input is not None)

# ── Process Button ──
if has_input:
    process_btn = st.button(
        "🚀  Process Command",
        type="primary",
        use_container_width=True,
        key="process_btn"
    )

    if process_btn:
        st.session_state.audio_processed = True
        st.session_state.pipeline_step = 1
        st.session_state.current_transcription = None
        st.session_state.current_intents = None
        st.session_state.current_results = None
        st.session_state.pending_confirmations = []
        st.session_state.benchmarks = {}

        # Decide if we need STT or can skip it (text prompt only mode)
        use_text_only = (text_prompt_input and text_prompt_input.strip() and not audio_file_path)

        if use_text_only:
            # ── Text-Only Mode: Skip STT ──
            st.markdown('<div class="section-label">STEP 2</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">✍️ Text Input (STT Skipped)</div>', unsafe_allow_html=True)

            stt_result = {
                "text": text_prompt_input.strip(),
                "segments": [],
                "language": "en",
                "language_name": "Text Input",
                "duration": 0.0,
                "confidence": 1.0,
                "processing_time": 0.0,
                "success": True,
                "error": None,
            }
            st.session_state.benchmarks["stt_time"] = 0.0

            st.markdown(f"""
            <div class="glass-card-success">
                <div class="section-label">Your Prompt</div>
                <p style="font-size:1.1rem;color:#E6EDF3;line-height:1.6;margin:8px 0;">
                    "{stt_result['text']}"
                </p>
                <div style="margin-top:8px;">
                    <span style="font-size:0.75rem;color:#4ECDC4;font-weight:600;">
                        ✍️ Direct text input — STT skipped
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Audio Mode: Run STT ──
            st.markdown('<div class="section-label">STEP 2</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🗣️ Speech-to-Text Transcription</div>', unsafe_allow_html=True)

            with st.spinner("🗣️ Transcribing audio with Whisper..."):
                stt_result = st.session_state.stt_engine.transcribe(
                    audio_file_path,
                    language=selected_language
                )

            # If user also provided a text prompt, combine it with the transcription
            if text_prompt_input and text_prompt_input.strip() and stt_result["success"]:
                combined_text = f"{stt_result['text']}\n\nUser Instructions: {text_prompt_input.strip()}"
                stt_result["text"] = combined_text

        st.session_state.benchmarks["stt_time"] = stt_result["processing_time"]

        if stt_result["success"] and stt_result["text"]:
            st.session_state.current_transcription = stt_result
            st.session_state.pipeline_step = 2

            # Display transcription result (only for audio mode, text prompt already shown above)
            if not use_text_only:
                langs_info = stt_result.get('languages_summary', f"{stt_result['language_name']} ({stt_result['language']})")
                st.markdown(f"""
                <div class="glass-card-success">
                    <div class="section-label">Transcription Result</div>
                    <p style="font-size:1.1rem;color:#E6EDF3;line-height:1.6;margin:8px 0;">
                        "{stt_result['text']}"
                    </p>
                    <div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;">
                        <span style="font-size:0.75rem;color:#4ECDC4;font-weight:600;">
                            🌐 {langs_info}
                        </span>
                        <span style="font-size:0.75rem;color:#8B949E;font-weight:600;">
                            🎯 Confidence: {round(stt_result['confidence'] * 100)}%
                        </span>
                        <span style="font-size:0.75rem;color:#8B949E;font-weight:600;">
                            ⏱ {stt_result['processing_time']}s
                        </span>
                        <span style="font-size:0.75rem;color:#8B949E;font-weight:600;">
                            ⏳ Duration: {stt_result['duration']}s
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Low confidence warning (Graceful Degradation)
            if stt_result["confidence"] < 0.5:
                st.warning("⚠️ Low transcription confidence. The audio might be unclear or in an unsupported language. Please try re-recording.")

            # ── STEP 3: Intent Classification ──
            st.markdown('<div class="section-label">STEP 3</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🧠 Intent Classification</div>', unsafe_allow_html=True)

            with st.spinner("🧠 Classifying intent with Ollama..."):
                st.session_state.intent_engine.model = st.session_state.ollama_model
                conversation_ctx = st.session_state.memory.get_conversation_context()
                intent_result = st.session_state.intent_engine.classify(
                    stt_result["text"],
                    conversation_history=conversation_ctx
                )

            st.session_state.benchmarks["intent_time"] = intent_result["processing_time"]

            if intent_result["success"]:
                st.session_state.current_intents = intent_result
                st.session_state.pipeline_step = 3

                # Display detected intents
                badges_html = "".join(
                    get_intent_badge(i["type"]) for i in intent_result["intents"]
                )
                compound_label = " <span style='color:#FFC107;font-size:0.75rem;font-weight:600;'>🔗 COMPOUND COMMAND</span>" if len(intent_result["intents"]) > 1 else ""

                st.markdown(f"""
                <div class="glass-card">
                    <div class="section-label">Detected Intents{compound_label}</div>
                    <div style="margin: 12px 0;">{badges_html}</div>
                    <p style="font-size:0.85rem;color:#8B949E;margin-top:8px;line-height:1.5;">
                        💭 {intent_result['reasoning']}
                    </p>
                    <span style="font-size:0.75rem;color:#8B949E;font-weight:600;">⏱ {intent_result['processing_time']}s</span>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="glass-card-warning">
                    <div class="section-label">⚠️ Intent Classification Issue</div>
                    <p style="color:#FFC107;font-size:0.9rem;">{intent_result['error']}</p>
                    <p style="color:#8B949E;font-size:0.85rem;">Falling back to general chat mode.</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.current_intents = intent_result

            # ── STEP 4: Tool Execution ──
            st.markdown('<div class="section-label">STEP 4</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚡ Tool Execution</div>', unsafe_allow_html=True)

            intents = intent_result.get("intents", [])
            results = []
            needs_confirm = []
            non_confirm = []

            # Separate intents that need confirmation from those that don't
            for intent in intents:
                if needs_confirmation(intent.get("type", "")):
                    needs_confirm.append(intent)
                else:
                    non_confirm.append(intent)

            # Execute non-confirmation intents immediately
            tool_start = time.time()
            for intent in non_confirm:
                with st.spinner(f"Executing: {intent['type'].replace('_', ' ').title()}..."):
                    result = execute_intent(
                        intent,
                        stt_result["text"],
                        st.session_state.ollama_model
                    )
                    results.append(result)

            # Handle intents that require confirmation (Human-in-the-Loop)
            if needs_confirm:
                # Pipe results from previous intents into confirmation intents
                enrich_intents_with_results(needs_confirm, results)

                # De-duplicate intents for the same file. LLM sometimes outputs BOTH create_file & write_code for the same file.
                seen_files = {}
                deduped_confirm = []
                for intent in needs_confirm:
                    params = intent.get("params", {})
                    fname = params.get("filename") or params.get("description")
                    if not fname:
                        deduped_confirm.append(intent)
                        continue
                    
                    if fname not in seen_files:
                        seen_files[fname] = intent
                        deduped_confirm.append(intent)
                    else:
                        # Prefer create_file over write_code if there are multiple intents for the same file
                        existing = seen_files[fname]
                        if existing["type"] == "write_code" and intent["type"] == "create_file":
                            idx = deduped_confirm.index(existing)
                            deduped_confirm[idx] = intent
                            seen_files[fname] = intent

                needs_confirm = deduped_confirm

                # Save to session state so confirmation persists across Streamlit re-runs
                st.session_state.pending_confirmations = needs_confirm
                st.session_state.confirm_context = {
                    "transcription": stt_result["text"],
                    "model": st.session_state.ollama_model,
                    "results": list(results),
                    "intent_result": intent_result,
                    "stt_result": stt_result,
                    "audio_file_path": audio_file_path,
                    "benchmarks": dict(st.session_state.benchmarks),
                }
                # Stop here — the persistent confirmation block below will handle the UI
                st.rerun()
            else:
                st.session_state.pipeline_step = 4

            tool_time = round(time.time() - tool_start, 2)
            st.session_state.benchmarks["tool_time"] = tool_time
            st.session_state.current_results = results

            # ── Display Results ──
            if results:
                st.markdown('<div class="section-label">RESULTS</div>', unsafe_allow_html=True)

                for result in results:
                    if not isinstance(result, dict):
                        continue

                    action = result.get("action", "unknown")
                    success = result.get("success", False)

                    if success:
                        if action == "general_chat":
                            st.markdown(f"""
                            <div class="glass-card-success">
                                {get_intent_badge('general_chat')}
                                <p style="font-size:1rem;color:#E6EDF3;margin-top:12px;line-height:1.7;">
                                    {result.get('response', '')}
                                </p>
                                <span style="font-size:0.75rem;color:#8B949E;">⏱ {result.get('processing_time', 0)}s</span>
                            </div>
                            """, unsafe_allow_html=True)

                        elif action == "summarize":
                            st.markdown(f"""
                            <div class="glass-card-success">
                                {get_intent_badge('summarize')}
                                <p style="font-size:1rem;color:#E6EDF3;margin-top:12px;line-height:1.7;">
                                    {result.get('summary', '')}
                                </p>
                                <div style="margin-top:8px;font-size:0.75rem;color:#8B949E;">
                                    📊 Compression: {result.get('original_length', 0)} → {result.get('summary_length', 0)} chars
                                    ({round(result.get('compression_ratio', 0) * 100)}%) |
                                    ⏱ {result.get('processing_time', 0)}s
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        elif action in ("create_file", "create_folder"):
                            st.markdown(f"""
                            <div class="glass-card-success">
                                {get_intent_badge('create_file')}
                                <div style="margin-top:12px;">
                                    <div class="file-tree-item">
                                        {'📁' if action == 'create_folder' else '📄'} {result.get('filename', '')}
                                    </div>
                                    <p style="font-size:0.8rem;color:#8B949E;margin-top:8px;">
                                        📂 Saved to: <code>{result.get('file_path', '')}</code>
                                    </p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        elif action == "write_code":
                            st.markdown(f"""
                            <div class="glass-card-success">
                                {get_intent_badge('write_code')}
                                <div style="margin-top:8px;">
                                    <div class="file-tree-item">
                                        💻 {result.get('filename', '')} ({result.get('language', '')})
                                    </div>
                                    <p style="font-size:0.8rem;color:#8B949E;margin-top:4px;">
                                        📂 Saved to: <code>{result.get('file_path', '')}</code> |
                                        📦 {result.get('size_bytes', 0)} bytes |
                                        ⏱ {result.get('processing_time', 0)}s
                                    </p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("👁️ View Generated Code"):
                                st.code(result.get("code", ""), language=result.get("language", "python"))

                    else:
                        error_msg = result.get("error", "Unknown error occurred.")
                        st.markdown(f"""
                        <div class="glass-card-danger">
                            <div class="section-label" style="color:#FF5252;">❌ Error</div>
                            <p style="color:#FF5252;font-size:0.9rem;">{error_msg}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # ── Benchmarks ──
            if st.session_state.benchmarks:
                st.markdown('<div class="section-label">⏱ PERFORMANCE BENCHMARKS</div>', unsafe_allow_html=True)
                render_benchmarks(st.session_state.benchmarks)

            # ── Save to Memory (only when no pending confirmations) ──
            if not st.session_state.pending_confirmations:
                intent_list = intent_result.get("intents", [])
                all_success = all(r.get("success", False) for r in results if isinstance(r, dict))
                st.session_state.memory.add_entry(
                    audio_file=audio_file_path,
                    transcription=stt_result["text"],
                    language=stt_result.get("language_name", ""),
                    intents=intent_list,
                    results=results,
                    status="completed" if all_success else "error",
                    benchmarks=st.session_state.benchmarks,
                )

            st.session_state.pipeline_step = 4

        else:
            # Transcription failed (Graceful Degradation)
            error_msg = stt_result.get("error", "Unknown transcription error.")
            st.markdown(f"""
            <div class="glass-card-danger">
                <div class="section-label" style="color:#FF5252;">❌ Transcription Failed</div>
                <p style="color:#FF5252;font-size:0.9rem;">{error_msg}</p>
                <p style="color:#8B949E;font-size:0.85rem;margin-top:8px;">
                    💡 Suggestions:<br>
                    • Ensure the audio is clear and not too noisy<br>
                    • Try selecting a specific language instead of Auto Detect<br>
                    • Use a larger Whisper model (small/medium) for better accuracy<br>
                    • Check that the audio file is not corrupted
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.memory.add_entry(
                audio_file=audio_file_path,
                transcription="[Transcription failed]",
                language="",
                intents=[],
                results=[],
                status="error",
                benchmarks=st.session_state.benchmarks,
            )

# ─────────────────────────────────────────────
# Human-in-the-Loop Confirmation (persists across re-runs)
# ─────────────────────────────────────────────
if st.session_state.pending_confirmations:
    st.divider()
    st.markdown('<div class="section-label">STEP 4 — CONFIRMATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">✋ Human-in-the-Loop Approval</div>', unsafe_allow_html=True)

    pending = st.session_state.pending_confirmations
    ctx = st.session_state.confirm_context

    st.markdown("""
    <div class="confirm-card">
        <div class="section-label" style="color:#FFC107;">⚠️ Confirmation Required</div>
        <p style="color:#E6EDF3;font-size:0.9rem;">The following actions require your approval before execution:</p>
    </div>
    """, unsafe_allow_html=True)

    for idx, intent in enumerate(pending):
        intent_type = intent.get("type", "unknown")
        params = intent.get("params", {})
        content_preview = params.get("content", "")[:200]

        st.markdown(f"""
        <div class="glass-card-warning">
            {get_intent_badge(intent_type)}
            <p style="color:#E6EDF3;font-size:0.85rem;margin-top:8px;">
                <strong>Action:</strong> {intent_type.replace('_', ' ').title()}<br>
                <strong>File:</strong> {params.get('filename', params.get('description', 'N/A'))}<br>
                <strong>Content:</strong> {content_preview if content_preview else '(empty file)'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Approve & Execute", type="primary", use_container_width=True, key="approve_confirm_btn"):
            confirm_results = list(ctx.get("results", []))
            tool_start = time.time()
            for intent in pending:
                with st.spinner(f"Executing: {intent['type'].replace('_', ' ').title()}..."):
                    result = execute_intent(
                        intent,
                        ctx["transcription"],
                        ctx["model"]
                    )
                    confirm_results.append(result)

            tool_time = round(time.time() - tool_start, 2)
            benchmarks = ctx.get("benchmarks", {})
            benchmarks["tool_time"] = benchmarks.get("tool_time", 0) + tool_time

            # Save to memory
            stt_ctx = ctx.get("stt_result", {})
            intent_ctx = ctx.get("intent_result", {})
            all_success = all(r.get("success", False) for r in confirm_results if isinstance(r, dict))
            st.session_state.memory.add_entry(
                audio_file=ctx.get("audio_file_path", ""),
                transcription=ctx.get("transcription", ""),
                language=stt_ctx.get("language_name", ""),
                intents=intent_ctx.get("intents", []),
                results=confirm_results,
                status="completed" if all_success else "error",
                benchmarks=benchmarks,
            )

            # Clear pending state
            st.session_state.pending_confirmations = []
            st.session_state.confirm_context = {}
            st.session_state.current_results = confirm_results
            st.session_state.pipeline_step = 4
            st.rerun()

    with col_no:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_confirm_btn"):
            st.session_state.pending_confirmations = []
            st.session_state.confirm_context = {}
            st.warning("❌ Operation cancelled by user.")
            st.rerun()

# ─────────────────────────────────────────────
# Output Files Section
# ─────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-label">OUTPUT FILES</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">📂 Generated Files</div>', unsafe_allow_html=True)

output_files = list_output_files()
if output_files:
    for f in output_files:
        icon = "📁" if f["is_dir"] else "📄"
        size = f"({f['size']} bytes)" if not f["is_dir"] else ""
        st.markdown(f"""
        <div class="file-tree-item">
            {icon} {f['name']} <span style="color:#8B949E;font-size:0.75rem;">{size}</span>
        </div>
        """, unsafe_allow_html=True)

    # Option to view or delete file contents
    file_names = [f["name"] for f in output_files if not f["is_dir"]]
    if file_names:
        col_sel, col_del = st.columns([3, 1])
        with col_sel:
            selected_file = st.selectbox("Preview a file:", ["Select..."] + file_names, key="file_preview")
        with col_del:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)  # Align with selectbox
            if selected_file != "Select..." and st.button("🗑️ Delete Data", use_container_width=True):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, selected_file))
                    st.success(f"Deleted {selected_file}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete {selected_file}: {e}")

        if selected_file != "Select...":
            fpath = os.path.join(OUTPUT_DIR, selected_file)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Detect language from extension
                ext = os.path.splitext(selected_file)[1].lower()
                lang_map = {
                    ".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".html": "html", ".css": "css", ".java": "java",
                    ".cpp": "cpp", ".c": "c", ".go": "go", ".rs": "rust",
                    ".rb": "ruby", ".php": "php", ".sh": "bash", ".sql": "sql",
                }
                st.code(content, language=lang_map.get(ext, "text"))
            except Exception as e:
                st.error(f"Cannot read file: {e}")
else:
    st.caption("No files generated yet. Process a voice command to create files.")

# ─────────────────────────────────────────────
# Benchmarking Dashboard Section
# ─────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-label">MODEL BENCHMARKING</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Performance Dashboard</div>', unsafe_allow_html=True)

stats = st.session_state.memory.get_total_stats()
if stats["total_commands"] > 0:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_commands']}</div>
            <div class="metric-label">Total Commands</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_stt_time']}s</div>
            <div class="metric-label">Avg STT Time</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_intent_time']}s</div>
            <div class="metric-label">Avg Intent Time</div>
        </div>
        """, unsafe_allow_html=True)
    with b4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_tool_time']}s</div>
            <div class="metric-label">Avg Tool Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-card" style="margin-top:16px;">
        <div class="section-label">Model Configuration</div>
        <p style="font-size:0.85rem;color:#E6EDF3;line-height:1.8;">
            🗣️ <strong>STT Model:</strong> faster-whisper (<code>{whisper_size}</code>) — CPU, INT8 quantized<br>
            🧠 <strong>LLM Model:</strong> Ollama (<code>{st.session_state.ollama_model}</code>) — Local inference<br>
            📊 <strong>Total Processing Time:</strong> {stats['total_processing_time']}s across {stats['total_commands']} commands<br>
            📁 <strong>Files Generated:</strong> {stats['files_created'] + stats['code_generated']} |
            📝 <strong>Summaries:</strong> {stats['summaries']} |
            💬 <strong>Chats:</strong> {stats['chats']}
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.caption("Process some commands to see benchmarking data here.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;padding:24px 0;">
    <p style="font-size:0.8rem;color:#8B949E;">
        Built with ❤️ using <strong>Streamlit</strong> • <strong>faster-whisper</strong> • <strong>Ollama</strong>
    </p>
    <p style="font-size:0.7rem;color:#484F58;">
        Mem0 AI/ML & Generative AI Developer Intern Assignment
    </p>
</div>
""", unsafe_allow_html=True)
