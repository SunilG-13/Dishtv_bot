"""
Voice Service Module
Handles Speech-to-Text (STT) and Text-to-Speech (TTS) for the 6 supported Indian languages.
Provides clean speech synthesis using gTTS with persistent caching and audio post-processing.
"""
import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from gtts import gTTS

from config import AUDIO_CACHE_DIR, SUPPORTED_LANGUAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_text_for_speech(text: str, max_chars: int = 350) -> str:
    """
    Cleans markdown formatting, links, URLs, and punctuation artifacts
    so synthesized speech sounds fluent and natural.
    """
    if not text:
        return ""

    # Remove URLs
    cleaned = re.sub(r'https?://\S+', '', text)
    # Convert markdown links [Label](url) to just Label
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
    # Remove markdown headers (#, ##), bold (**), italic (*), code (`)
    cleaned = re.sub(r'[*_#`~>]', '', cleaned)
    # Remove bullet list indicators
    cleaned = re.sub(r'^\s*[-•*+]\s+', '', cleaned, flags=re.MULTILINE)
    # Remove numbered list markers like 1., 2.
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)
    # Remove citation chips or bracket references [1]
    cleaned = re.sub(r'\[\d+\]', '', cleaned)
    # Normalize spaces and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if len(cleaned) <= max_chars:
        return cleaned

    # Truncate cleanly at a sentence or clause boundary
    candidate = cleaned[:max_chars]
    for delimiter in ['. ', '! ', '? ', '। ', '\n', '.']:
        idx = candidate.rfind(delimiter)
        if idx > 100:
            return candidate[:idx + len(delimiter)].strip()

    return candidate.strip()


class VoiceService:
    """Manages audio generation, caching, speech conversion, and Whisper STT."""

    def __init__(self, cache_dir: Path = AUDIO_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._whisper_model = None

    def get_whisper_model(self):
        """Loads and caches the OpenAI Whisper multilingual speech model ('small' for high accuracy)."""
        if self._whisper_model is None:
            import whisper
            logger.info("Loading OpenAI Whisper 'small' model for high-accuracy multilingual speech recognition...")
            self._whisper_model = whisper.load_model("small")
            logger.info("OpenAI Whisper 'small' model loaded successfully.")
        return self._whisper_model

    def transcribe_audio_file(self, audio_file_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribes speech from an audio file using OpenAI Whisper model.
        Restricts language candidates to English + 5 Indian languages (hi, te, kn, ta, gu).
        Suppresses repetition loops, hallucinatory patterns, and empty background noises.
        """
        try:
            import whisper
            import numpy as np
            model = self.get_whisper_model()

            # 1. Load audio and measure RMS volume to reject absolute silence
            audio = whisper.load_audio(str(audio_file_path))
            if len(audio) == 0:
                return {"success": False, "text": "", "error": "Empty audio"}

            # Calculate peak amplitude and RMS energy to reject pure silence or clicks
            peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
            logger.info(f"Audio duration: {len(audio)/16000:.2f}s, Peak: {peak:.4f}, RMS: {rms:.4f}")
            if (peak < 0.008 and rms < 0.002) and len(audio) > 8000:
                logger.warning("Audio amplitude below audible threshold (pure silence/background click).")
                return {"success": False, "text": "", "error": "No audible speech detected"}

            # 2. Restrict language detection to the 6 supported languages
            target_lang = None
            if language and language in ["en", "hi", "te", "kn", "ta", "gu"]:
                target_lang = language
            else:
                audio_padded = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio_padded, n_mels=model.dims.n_mels).to(model.device)
                _, probs = model.detect_language(mel)
                allowed_langs = ["en", "hi", "te", "kn", "ta", "gu"]
                filtered_probs = {l: probs.get(l, 0.0) for l in allowed_langs}
                best_lang = max(filtered_probs, key=filtered_probs.get)
                target_lang = best_lang if filtered_probs[best_lang] >= 0.01 else "en"
                logger.info(f"Constrained language detection: {target_lang} (confidence: {filtered_probs.get(target_lang, 0.0):.3f})")

            # 3. Transcribe with clean parameters (no hallucination prompt, fast greedy decode)
            decode_options = {
                "fp16": False,
                "language": target_lang,
                "task": "transcribe",
                "condition_on_previous_text": False,
                "temperature": 0.0,
                "best_of": 1,
                "beam_size": 1,
                "no_speech_threshold": 0.6,
                "compression_ratio_threshold": 2.4,
            }

            logger.info(f"Transcribing audio file with Whisper 'small': {audio_file_path.name}")
            result = model.transcribe(str(audio_file_path), **decode_options)
            raw_text = result.get("text", "").strip()

            # Clean punctuation-only artifacts (e.g. '.', '...', '!')
            clean_check = raw_text.strip(" .?!,;:-_~`'\"/\\")
            if not clean_check:
                logger.warning("Whisper produced empty or punctuation-only output.")
                return {"success": False, "text": "", "error": "No audible speech detected"}

            # 4. Anti-repetition check (discards repeated phrase loops like '1.5kg 2.5kg...')
            words = raw_text.split()
            if len(words) >= 4:
                is_repetitive = False
                for w in set(words):
                    if words.count(w) >= 3 and len(words) <= words.count(w) * 1.6:
                        is_repetitive = True
                        break
                if is_repetitive:
                    logger.warning(f"Discarding repetitive transcription loop: '{raw_text}'")
                    return {"success": False, "text": "", "error": "Repetitive speech artifact"}

            logger.info(f"Whisper transcription result: '{raw_text}' (language: {target_lang})")
            return {
                "success": True,
                "text": raw_text,
                "language": target_lang
            }
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }

    def synthesize_speech_file(self, text: str, lang: str = "en") -> Optional[Path]:
        """
        Synthesizes text to speech using gTTS and caches the mp3 file.
        Returns the Path to the generated audio file.
        """
        if not text or not text.strip():
            return None

        # Clean markdown artifacts for natural listening
        spoken_text = clean_text_for_speech(text)
        if not spoken_text:
            return None

        # Determine language code (en, hi, kn, te, ta, gu)
        lang_key = lang.lower() if lang else "en"
        lang_info = SUPPORTED_LANGUAGES.get(lang_key, SUPPORTED_LANGUAGES["en"])
        gtts_code = lang_info.get("gtts_code", "en")

        # Generate unique cache filename based on text hash and language
        text_hash = hashlib.md5(f"{gtts_code}_{spoken_text}".encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"tts_{gtts_code}_{text_hash}.mp3"

        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path

        try:
            logger.info(f"Generating TTS audio for [{gtts_code}], text length: {len(spoken_text)} chars...")
            tts = gTTS(text=spoken_text, lang=gtts_code, slow=False)
            tts.save(str(cache_path))
            return cache_path
        except Exception as e:
            logger.warning(f"gTTS error for language {gtts_code}: {e}. Retrying with English fallback...")
            try:
                tts = gTTS(text=spoken_text, lang="en", slow=False)
                tts.save(str(cache_path))
                return cache_path
            except Exception as e_fallback:
                logger.error(f"Failed fallback TTS: {e_fallback}")
                return None

    def get_audio_path_by_filename(self, filename: str) -> Optional[Path]:
        """Safely fetches a cached audio file by filename without path traversal."""
        safe_name = os.path.basename(filename)
        target = self.cache_dir / safe_name
        if target.exists() and target.is_file():
            return target
        return None


# Singleton instance
_voice_service_instance = None

def get_voice_service() -> VoiceService:
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance
