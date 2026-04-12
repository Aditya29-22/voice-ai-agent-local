"""
Memory Module
Maintains persistent session history for conversation context and action tracking.
"""

import time
from datetime import datetime


class SessionMemory:
    """Manages session-level memory for the AI agent."""

    def __init__(self):
        self.history = []
        self.created_at = datetime.now().isoformat()

    def add_entry(
        self,
        audio_file: str = "",
        transcription: str = "",
        language: str = "",
        intents: list = None,
        results: list = None,
        status: str = "completed",
        benchmarks: dict = None,
    ):
        """Add a new entry to the session history."""
        entry = {
            "id": len(self.history) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "datetime": datetime.now().isoformat(),
            "audio_file": audio_file,
            "transcription": transcription,
            "language": language,
            "intents": intents or [],
            "results": results or [],
            "status": status,
            "benchmarks": benchmarks or {},
        }
        self.history.append(entry)
        return entry

    def get_history(self) -> list:
        """Get the full session history."""
        return self.history

    def get_recent(self, n: int = 5) -> list:
        """Get the last N entries."""
        return self.history[-n:]

    def get_conversation_context(self) -> list:
        """
        Get conversation history formatted for LLM context.
        Returns a list of simplified entries for the intent classifier.
        """
        context = []
        for entry in self.history[-5:]:
            ctx = {
                "transcription": entry.get("transcription", ""),
                "result": "",
            }
            # Summarize results for context
            for result in entry.get("results", []):
                if isinstance(result, dict):
                    if result.get("action") == "general_chat":
                        ctx["result"] = result.get("response", "")[:200]
                    elif result.get("action") == "summarize":
                        ctx["result"] = result.get("summary", "")[:200]
                    elif result.get("action") in ("create_file", "write_code"):
                        fname = result.get('filename', '')
                        lang = result.get('language', '')
                        action_label = "Generated code file" if result.get("action") == "write_code" else "Created file"
                        desc = f"{action_label}: {fname}"
                        if lang:
                            desc += f" (language: {lang})"
                        if result.get("code"):
                            desc += f" | Code preview: {result['code'][:150]}"
                        ctx["result"] = desc
            context.append(ctx)
        return context

    def get_total_stats(self) -> dict:
        """Get aggregated statistics for the session."""
        total_entries = len(self.history)
        total_files_created = 0
        total_code_generated = 0
        total_summaries = 0
        total_chats = 0
        total_stt_time = 0.0
        total_intent_time = 0.0
        total_tool_time = 0.0

        for entry in self.history:
            benchmarks = entry.get("benchmarks", {})
            total_stt_time += benchmarks.get("stt_time", 0)
            total_intent_time += benchmarks.get("intent_time", 0)
            total_tool_time += benchmarks.get("tool_time", 0)

            for result in entry.get("results", []):
                if isinstance(result, dict):
                    action = result.get("action", "")
                    if action == "create_file" or action == "create_folder":
                        total_files_created += 1
                    elif action == "write_code":
                        total_code_generated += 1
                    elif action == "summarize":
                        total_summaries += 1
                    elif action == "general_chat":
                        total_chats += 1

        return {
            "total_commands": total_entries,
            "files_created": total_files_created,
            "code_generated": total_code_generated,
            "summaries": total_summaries,
            "chats": total_chats,
            "avg_stt_time": round(total_stt_time / max(total_entries, 1), 2),
            "avg_intent_time": round(total_intent_time / max(total_entries, 1), 2),
            "avg_tool_time": round(total_tool_time / max(total_entries, 1), 2),
            "total_processing_time": round(total_stt_time + total_intent_time + total_tool_time, 2),
        }

    def clear(self):
        """Clear all session history."""
        self.history = []
