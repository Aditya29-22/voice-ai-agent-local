"""
Tool Execution Module
Handles file operations, code generation, text summarization, and chat.
All file operations are sandboxed to the output/ directory.
"""

import os
import time
import ollama


# Safety: All file operations are restricted to this directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def _ensure_output_dir():
    """Ensure the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_path(filename: str) -> str:
    """
    Validate and return a safe file path within output/ directory.
    Prevents directory traversal attacks.
    """
    _ensure_output_dir()
    # Strip any path components that could escape the sandbox
    safe_name = os.path.basename(filename)
    if not safe_name:
        safe_name = "untitled.txt"
    full_path = os.path.join(OUTPUT_DIR, safe_name)
    # Double-check the resolved path is within OUTPUT_DIR
    resolved = os.path.realpath(full_path)
    if not resolved.startswith(os.path.realpath(OUTPUT_DIR)):
        raise ValueError(f"Security error: Path '{filename}' escapes the output directory.")
    return full_path


def create_file(filename: str, content: str = "") -> dict:
    """
    Create a file in the output/ directory.

    Args:
        filename: Name of the file to create
        content: Optional content to write

    Returns:
        dict with result details
    """
    start_time = time.time()
    try:
        file_path = _safe_path(filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "action": "create_file",
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "size_bytes": os.path.getsize(file_path),
            "content_preview": content[:200] if content else "(empty file)",
            "processing_time": round(time.time() - start_time, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": "create_file",
            "file_path": "",
            "filename": filename,
            "size_bytes": 0,
            "content_preview": "",
            "processing_time": round(time.time() - start_time, 2),
            "error": str(e),
        }


def create_folder(foldername: str) -> dict:
    """
    Create a folder inside the output/ directory.
    """
    start_time = time.time()
    try:
        _ensure_output_dir()
        safe_name = os.path.basename(foldername)
        if not safe_name:
            safe_name = "new_folder"
        folder_path = os.path.join(OUTPUT_DIR, safe_name)
        os.makedirs(folder_path, exist_ok=True)

        return {
            "success": True,
            "action": "create_folder",
            "file_path": folder_path,
            "filename": safe_name,
            "processing_time": round(time.time() - start_time, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "action": "create_folder",
            "file_path": "",
            "filename": foldername,
            "processing_time": round(time.time() - start_time, 2),
            "error": str(e),
        }


def generate_code(description: str, language: str = "python", filename: str = None, model: str = "mistral") -> dict:
    """
    Generate code using Ollama and save to a file in output/.

    Args:
        description: What the code should do
        language: Programming language
        filename: Target filename (auto-generated if None)
        model: Ollama model to use

    Returns:
        dict with generated code and file details
    """
    start_time = time.time()

    # Auto-generate filename if not provided
    if not filename:
        ext_map = {
            "python": ".py", "javascript": ".js", "typescript": ".ts",
            "java": ".java", "c": ".c", "cpp": ".cpp", "c++": ".cpp",
            "html": ".html", "css": ".css", "go": ".go", "rust": ".rs",
            "ruby": ".rb", "php": ".php", "swift": ".swift", "kotlin": ".kt",
            "bash": ".sh", "shell": ".sh", "sql": ".sql",
        }
        ext = ext_map.get(language.lower(), ".txt")
        safe_desc = "".join(c if c.isalnum() else "_" for c in description[:30]).strip("_").lower()
        filename = f"{safe_desc}{ext}"

    try:
        prompt = f"""Generate clean, well-commented {language} code for the following task:

{description}

RULES:
- Return ONLY the code, no explanations or markdown formatting.
- Include helpful comments in the code.
- Make the code production-ready and well-structured.
- Do NOT wrap the code in markdown code blocks (no ``` markers).
"""

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": f"You are an expert {language} developer. Generate clean, production-ready code. Return ONLY the code with no markdown formatting."},
                {"role": "user", "content": prompt}
            ]
        )

        code = response["message"]["content"]

        # Clean up: remove markdown code blocks if LLM included them anyway
        code = _clean_code_block(code)

        # Save to file
        file_result = create_file(filename, code)

        return {
            "success": file_result["success"],
            "action": "write_code",
            "code": code,
            "language": language,
            "file_path": file_result.get("file_path", ""),
            "filename": file_result.get("filename", filename),
            "size_bytes": file_result.get("size_bytes", 0),
            "processing_time": round(time.time() - start_time, 2),
            "error": file_result.get("error"),
        }

    except Exception as e:
        return {
            "success": False,
            "action": "write_code",
            "code": "",
            "language": language,
            "file_path": "",
            "filename": filename,
            "size_bytes": 0,
            "processing_time": round(time.time() - start_time, 2),
            "error": str(e),
        }


def summarize_text(text: str, model: str = "mistral") -> dict:
    """
    Summarize the provided text using Ollama.
    """
    start_time = time.time()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise summarization assistant. Provide clear, concise summaries that capture the key points."},
                {"role": "user", "content": f"Please provide a clear and concise summary of the following text:\n\n{text}"}
            ]
        )

        summary = response["message"]["content"]

        return {
            "success": True,
            "action": "summarize",
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / max(len(text), 1), 2),
            "processing_time": round(time.time() - start_time, 2),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "action": "summarize",
            "summary": "",
            "original_length": len(text),
            "summary_length": 0,
            "compression_ratio": 0,
            "processing_time": round(time.time() - start_time, 2),
            "error": str(e),
        }


def general_chat(message: str, conversation_history: list = None, model: str = "mistral") -> dict:
    """
    Handle general chat using Ollama with conversation context.
    """
    start_time = time.time()

    try:
        messages = [
            {"role": "system", "content": "You are a helpful, friendly AI assistant. Provide clear and informative responses."}
        ]

        # Add conversation history for context
        if conversation_history:
            for entry in conversation_history[-5:]:
                if entry.get("transcription"):
                    messages.append({"role": "user", "content": entry["transcription"]})
                if entry.get("result"):
                    messages.append({"role": "assistant", "content": str(entry["result"])})

        messages.append({"role": "user", "content": message})

        response = ollama.chat(
            model=model,
            messages=messages,
        )

        reply = response["message"]["content"]

        return {
            "success": True,
            "action": "general_chat",
            "response": reply,
            "processing_time": round(time.time() - start_time, 2),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "action": "general_chat",
            "response": "",
            "processing_time": round(time.time() - start_time, 2),
            "error": str(e),
        }


def _clean_code_block(text: str) -> str:
    """Remove markdown code block markers from LLM output."""
    lines = text.strip().split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def list_output_files() -> list:
    """List all user-generated files in the output/ directory (hides dotfiles)."""
    _ensure_output_dir()
    files = []
    for item in os.listdir(OUTPUT_DIR):
        # Skip hidden/system files like .gitkeep
        if item.startswith("."):
            continue
        item_path = os.path.join(OUTPUT_DIR, item)
        files.append({
            "name": item,
            "path": item_path,
            "is_dir": os.path.isdir(item_path),
            "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0,
        })
    return files
