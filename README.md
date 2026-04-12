# 🎙️ Voice-Controlled Local AI Agent

> **Mem0 AI/ML & Generative AI Developer Intern Assignment**
## 📖 Project Overview

A fully local AI agent that accepts audio input (microphone or file upload), transcribes speech to text using **faster-whisper**, classifies user intent using **Ollama LLM**, and executes appropriate tools — all displayed through a premium **Streamlit** interface with real-time pipeline visualization.

The primary goal of this repository is to demonstrate an end-to-end multi-modal pipeline executing entirely on local hardware, prioritizing data privacy, zero-latency network dependencies, and robust system architecture.

---

## ✨ Features

### Core Pipeline
| Step | Component | Technology |
|------|-----------|------------|
| 1️⃣ | **Audio Input** | Microphone recording + File upload (WAV, MP3, M4A, OGG, FLAC) |
| 2️⃣ | **Speech-to-Text** | faster-whisper (base model, CPU, INT8 quantized) |
| 3️⃣ | **Intent Classification** | Ollama LLM (Mistral) with structured JSON output |
| 4️⃣ | **Tool Execution** | File ops, code gen, summarize, chat — sandboxed to `output/` |
| 5️⃣ | **UI Display** | Streamlit with glassmorphism dark theme |

### Supported Intents
- 📁 **Create File** — Create files or folders in the `output/` directory
- 💻 **Write Code** — Generate production-ready code and save to file
- 📝 **Summarize** — Summarize provided text or transcribed content
- 💬 **General Chat** — Conversational AI with memory context

### 🏆 Bonus Features (All Implemented)
- 🔗 **Compound Commands** — Multiple intents in one audio input (e.g., "Summarize this and save it to summary.txt")
- ✋ **Human-in-the-Loop** — UI confirmation prompt before executing file operations
- 🛡️ **Graceful Degradation** — Smooth error handling for unclear audio, Ollama disconnection, unmapped intents
- 💾 **Memory** — Persistent session history with conversation context for better intent understanding
- 📊 **Model Benchmarking** — Real-time performance dashboard comparing STT, intent classification, and tool execution times

### 🌐 Multi-Language Support
Supports 25+ languages including: English, Hindi, Spanish, French, German, Japanese, Chinese, Korean, Arabic, Portuguese, Russian, Italian, Tamil, Telugu, Bengali, Urdu, Marathi, Gujarati, Kannada, Malayalam, and more — with automatic language detection.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│  │  Audio   │→ │   STT    │→ │  Intent   │→ │   Tool     │  │
│  │  Input   │  │ (Whisper)│  │ Classifier│  │ Execution  │  │
│  └─────────┘  └──────────┘  └───────────┘  └────────────┘  │
│       │              │             │              │          │
│       │              │             │              │          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Session Memory & Benchmarks             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   output/ folder   │
                    │  (sandboxed I/O)   │
                    └───────────────────┘
```

---

## 🚀 Setup Instructions

### Prerequisites
- **Python 3.10+** (tested with 3.14)
- **Ollama** installed and running locally
- A microphone (for live recording) or audio files for upload

### 1. Clone & Install

```bash
git clone <repo-url>
cd memo
pip install -r requirements.txt
```

### 2. Setup Ollama

```bash
# Install Ollama from https://ollama.com
# Start the Ollama server
ollama serve

# Pull the Mistral model (used for intent classification & tool execution)
ollama pull mistral
```

### 3. Run the Application

```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
memo/
├── .streamlit/
│   └── config.toml          # Dark theme configuration
├── agent/
│   ├── __init__.py           # Package init
│   ├── stt.py                # Speech-to-Text module (faster-whisper)
│   ├── intent.py             # Intent classification (Ollama + JSON parsing)
│   ├── tools.py              # Tool execution (file ops, code gen, summarize, chat)
│   └── memory.py             # Session memory & history tracking
├── output/                   # Sandboxed directory for generated files (safety constraint)
├── uploads/                  # Temporary audio file storage
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔐 Safety Constraints

- All file creation and code writing is **restricted to the `output/` folder** within the repository
- Directory traversal attacks are prevented via path validation
- File operations require **explicit user confirmation** before execution (Human-in-the-Loop)
- No files outside the sandbox can be read, modified, or deleted

---

## 🧠 Models Used & Hardware Architecture

To ensure local privacy, efficient resource usage, and offline capabilities, the system leverages optimized models:

| Component | Model & Choice | Implementation Details & Justification |
|-----------|----------------|----------------------------------------|
| **Speech-To-Text (STT)** | **faster-whisper** (Base/Small) | CTranslate2-based implementation allows for maximum CPU efficiency. INT8 quantization reduces VRAM/RAM constraints significantly while preserving transcription accuracy. |
| **Intent & Reasoning (LLM)** | **Ollama** (Mistral / Llama 3) | Mistral is utilized for its strong instruction-following capabilities in smaller parameter sizes (7B). Readily interchangeable with Llama 3 for improved performance. |
| **Hardware Constraint** | **CPU-Optimized Sandbox** | The architecture is specifically designed to fallback gracefully to CPU execution when CUDA/GPU is unavailable, demonstrating accessible AI deployment. |

---

## 📊 Model Benchmarking

The application includes a built-in benchmarking dashboard that tracks:
- **STT Processing Time** — Whisper transcription latency per audio input
- **Intent Classification Time** — Ollama LLM inference time for intent detection
- **Tool Execution Time** — Time to execute detected tools (file creation, code gen, etc.)
- **Aggregate Statistics** — Average times across all session commands

Performance comparisons across different model sizes (tiny/base/small/medium for Whisper, mistral/llama3 for Ollama) can be observed by switching models in the sidebar settings.

---

## 💡 Example Usage

1. **"Create a Python file with a retry function"**
   - Detects: `write_code` + `create_file` (compound)
   - Generates Python code with retry logic
   - Asks for confirmation, then saves to `output/`

2. **"Summarize this text and save it to summary.txt"**
   - Detects: `summarize` + `create_file` (compound)
   - Summarizes the transcription
   - Creates `output/summary.txt` with the summary

3. **"What is machine learning?"**
   - Detects: `general_chat`
   - Responds conversationally with context awareness

---

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** — Web UI framework
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — CTranslate2-based Whisper implementation for fast STT
- **[Ollama](https://ollama.com/)** — Local LLM inference (Mistral model)
- **Python OS module** — File system operations

---

## 🚧 Challenges Faced & Solutions

Building a fully local multi-modal agent presented several unique engineering challenges:

1. **Structured Tool Calling with Local LLMs:**
   - *Challenge:* Smaller local models (like Mistral-7B) often hallucinate or fail to adhere to strict JSON schemas needed for tool execution.
   - *Solution:* Implemented rigorous prompt engineering with few-shot examples and a robust parsing layer using Regex fallbacks to extract intents safely, even if the LLM output is malformed.

2. **Real-Time CPU Audio Processing:**
   - *Challenge:* Running standard Whisper models on a CPU causes severe latency, leading to a poor UX.
   - *Solution:* Integrated `faster-whisper` and applied INT8 quantization, resulting in a ~4x speedup on CPU inference without a discernible drop in Word Error Rate (WER).

3. **Compound User Commands:**
   - *Challenge:* Users naturally issue multi-part commands (e.g., "Summarize this and save it to a file"). 
   - *Solution:* Engineered the agent logic to dynamically classify arrays of intents, looping over them sequentially and seamlessly passing context between tool executions.

---

## 🔮 Future Roadmap

- **Streaming Transcriptions:** Implement WebSockets for real-time STT word-by-word streaming.
- **RAG Integration:** Allow the LLM to search local documents using vector embeddings (e.g., ChromaDB) for context-augmented chat.
- **Voice Response (TTS):** Integrate local SpeechT5 or Piper TTS for a fully two-way voice conversational agent.

---

## 📝 License

This project was built as part of the Mem0 AI/ML & Generative AI Developer Intern Assignment.
