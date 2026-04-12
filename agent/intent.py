"""
Intent Classification Module
Uses Ollama LLM to classify user intent from transcribed text.
Supports compound commands (multiple intents in one input).

Performance optimizations:
- Concise system prompt → fewer tokens → faster inference
- Temperature 0 → deterministic, no sampling overhead
- num_predict limit → prevents runaway generation
- Only last 3 turns of history → less context to process
"""

import json
import time
import re
import ollama


INTENT_SYSTEM_PROMPT = """Classify the user's command into intents. Return JSON only.

INTENTS:
- "create_file": Create a standard file or save summarized/chat text. Params: {"filename":"name.ext","file_type":"file"|"folder"}
- "write_code": ONLY for generating PROGRAMMING code (Python, Java, JS, etc). Params: {"filename":"name.ext","language":"<detected_language>","description":"what to do"}
- "summarize": Summarize text. Params: {"text_to_summarize":"the text"}
- "general_chat": Chat/question. Params: {"message":"user message"}

RULES:
- One input can have MULTIPLE intents (compound commands).
- CRITICAL: "Save this summary/text to a file" → summarize + create_file (NEVER write_code).
- CRITICAL: "Write the text to a file" → create_file (NEVER write_code).
- "Create a Python script with retry logic" → write_code (language="python", filename="retry.py").
- "Write a Java class for a Dog" → write_code (language="java", filename="Dog.java").
- If unsure, use general_chat.
- Extract right filenames, languages, descriptions from the command.

JSON FORMAT:
{"intents":[{"type":"intent_type","params":{...},"confidence":0.9}],"reasoning":"brief reason"}"""


class IntentClassifier:
    """Classifies user intent using Ollama LLM with optimized performance."""

    def __init__(self, model: str = "mistral"):
        self.model = model

    def classify(self, transcribed_text: str, conversation_history: list = None) -> dict:
        """
        Classify the intent of transcribed text.

        Args:
            transcribed_text: The text from speech-to-text
            conversation_history: Previous conversation for context

        Returns:
            dict with keys: intents, reasoning, processing_time, success, error
        """
        start_time = time.time()

        try:
            # Build messages — keep context minimal for speed
            messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]

            if conversation_history:
                # Only last 3 turns for speed (was 5)
                for entry in conversation_history[-3:]:
                    if entry.get("transcription"):
                        messages.append({
                            "role": "user",
                            "content": f"[Previous]: {entry['transcription'][:150]}"
                        })
                    if entry.get("result"):
                        messages.append({
                            "role": "assistant",
                            "content": f"[Result]: {str(entry['result'])[:300]}"
                        })

            messages.append({
                "role": "user",
                "content": f'Classify: "{transcribed_text}"'
            })

            response = ollama.chat(
                model=self.model,
                messages=messages,
                format="json",
                options={
                    "temperature": 0,        # Deterministic = faster
                    "num_predict": 300,       # Limit output tokens
                    "top_p": 0.9,
                },
            )

            response_text = response["message"]["content"]
            parsed = self._parse_response(response_text)

            processing_time = round(time.time() - start_time, 2)

            return {
                "intents": parsed.get("intents", []),
                "reasoning": parsed.get("reasoning", ""),
                "raw_response": response_text,
                "processing_time": processing_time,
                "success": True,
                "error": None,
            }

        except Exception as e:
            processing_time = round(time.time() - start_time, 2)

            # Graceful degradation: fall back to general_chat
            return {
                "intents": [{
                    "type": "general_chat",
                    "params": {"message": transcribed_text},
                    "confidence": 0.5,
                }],
                "reasoning": f"Fallback to general chat due to error: {str(e)}",
                "raw_response": "",
                "processing_time": processing_time,
                "success": False,
                "error": str(e),
            }

    def _parse_response(self, response_text: str) -> dict:
        """Parse the LLM JSON response, with fallback handling."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Ultimate fallback
            return {
                "intents": [{
                    "type": "general_chat",
                    "params": {"message": response_text},
                    "confidence": 0.3,
                }],
                "reasoning": "Could not parse LLM response, defaulting to general chat."
            }

    def check_connection(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models] if hasattr(models, 'models') else []
            return any(self.model in name for name in model_names)
        except Exception:
            return False
