"""
wav2vec_model.py — Wrapper for Wav2Vec2 / MMS STT models.

Supports three backends:
  - HuggingFace transformers (Wav2Vec2ForCTC) for standard wav2vec2 models
  - HuggingFace transformers (MMS) for facebook/mms-* multilingual models
  - SpeechBrain (EncoderASR) for speechbrain/* models

All backends provide a unified `transcribe(audio, language) -> str` interface.
"""

import logging
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ISO 639-1 (2-letter) → ISO 639-3 (3-letter) mapping for MMS models
_ISO1_TO_ISO3 = {
    "en": "eng", "tr": "tur", "de": "deu", "fr": "fra",
    "es": "spa", "it": "ita", "pt": "por", "nl": "nld",
    "ru": "rus", "zh": "cmn", "ja": "jpn", "ko": "kor",
    "ar": "ara", "hi": "hin", "pl": "pol", "sv": "swe",
    "uk": "ukr",
}


class Wav2VecModel:
    """
    Unified Wav2Vec2 wrapper supporting both HuggingFace and SpeechBrain backends.

    The backend is selected automatically based on the engine parameter:
      - "wav2vec"    → HuggingFace transformers (Wav2Vec2ForCTC + Wav2Vec2Processor)
      - "wav2vec-sb" → SpeechBrain (EncoderASR)
    """

    def __init__(self, model_id: str, device: str = "cpu", engine: str = "wav2vec"):
        """
        Load a Wav2Vec2 model.

        Args:
            model_id: HuggingFace model ID (e.g. "facebook/wav2vec2-base-960h").
            device: "cuda" or "cpu".
            engine: "wav2vec" for HuggingFace transformers, "wav2vec-sb" for SpeechBrain.
        """
        self.model_id = model_id
        self.device = device
        self.engine = engine
        self._available = False
        self._is_mms = "mms" in model_id.lower()

        # HuggingFace transformers objects
        self._hf_model = None
        self._hf_processor = None
        self._current_lang = None  # Tracks active MMS language adapter

        # SpeechBrain objects
        self._sb_model = None

        if engine == "wav2vec-sb":
            self._load_speechbrain(model_id, device)
        else:
            self._load_transformers(model_id, device)

    def _load_transformers(self, model_id: str, device: str):
        """Load model using HuggingFace transformers (supports standard + MMS)."""
        try:
            from transformers import Wav2Vec2ForCTC

            logger.info(f"Loading Wav2Vec2 '{model_id}' (transformers) on {device}...")

            # MMS models use AutoProcessor; standard models use Wav2Vec2Processor
            if self._is_mms:
                from transformers import AutoProcessor
                self._hf_processor = AutoProcessor.from_pretrained(model_id)
            else:
                from transformers import Wav2Vec2Processor
                self._hf_processor = Wav2Vec2Processor.from_pretrained(model_id)

            # Try standard loading first; on torch CVE restriction, try fallbacks
            try:
                self._hf_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
            except (ImportError, RuntimeError, ValueError, OSError) as e:
                err_msg = str(e).lower()
                if any(k in err_msg for k in ("vulnerability", "weights_only", "cve")):
                    # Tier 2: try safetensors format (bypasses torch.load entirely)
                    try:
                        logger.warning(
                            f"Retrying '{model_id}' with use_safetensors=True..."
                        )
                        self._hf_model = Wav2Vec2ForCTC.from_pretrained(
                            model_id, use_safetensors=True
                        ).to(device)
                    except Exception:
                        # Tier 3: temporarily allow weights_only=False
                        logger.warning(
                            f"Safetensors not available for '{model_id}', "
                            f"falling back to weights_only=False..."
                        )
                        original_torch_load = torch.load
                        def _permissive_load(*a, **kw):
                            kw["weights_only"] = False
                            return original_torch_load(*a, **kw)
                        torch.load = _permissive_load
                        try:
                            self._hf_model = Wav2Vec2ForCTC.from_pretrained(
                                model_id
                            ).to(device)
                        finally:
                            torch.load = original_torch_load
                else:
                    raise

            self._hf_model.eval()
            self._available = True
            logger.info(f"Wav2Vec2 '{model_id}' loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 '{model_id}': {e}")

    def _load_speechbrain(self, model_id: str, device: str):
        """Load model using SpeechBrain (with huggingface_hub compat fix)."""
        try:
            # Import SpeechBrain FIRST so all its submodules land in sys.modules
            from speechbrain.inference.ASR import EncoderASR

            # THEN patch every hf_hub_download reference inside speechbrain
            # and huggingface_hub — this fixes the 'use_auth_token' removal
            self._patch_hf_compat_all()

            logger.info(f"Loading Wav2Vec2 '{model_id}' (SpeechBrain) on {device}...")
            self._sb_model = EncoderASR.from_hparams(
                source=model_id,
                run_opts={"device": device},
            )
            self._available = True
            logger.info(f"Wav2Vec2 '{model_id}' (SpeechBrain) loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load SpeechBrain '{model_id}': {e}")

    @staticmethod
    def _patch_hf_compat_all():
        """
        Patch hf_hub_download / snapshot_download in **every** loaded module
        that references them (speechbrain.*, huggingface_hub, huggingface_hub.*).

        SpeechBrain copies the function reference via
        ``from huggingface_hub import hf_hub_download`` at module-level,
        so patching huggingface_hub alone is not enough — the local
        reference in speechbrain.utils.fetching (and others) must also
        be replaced.
        """
        try:
            for mod_name, mod in list(sys.modules.items()):
                if mod is None:
                    continue
                if not mod_name.startswith(("speechbrain", "huggingface_hub")):
                    continue

                for fn_name in ("hf_hub_download", "snapshot_download"):
                    fn = getattr(mod, fn_name, None)
                    if fn is None or not callable(fn):
                        continue
                    if getattr(fn, "_sb_compat_patched", False):
                        continue

                    def _make_wrapper(_orig):
                        def _wrapper(*args, **kwargs):
                            if "use_auth_token" in kwargs:
                                val = kwargs.pop("use_auth_token")
                                if "token" not in kwargs and val not in (None, False):
                                    kwargs["token"] = val
                            return _orig(*args, **kwargs)
                        _wrapper._sb_compat_patched = True
                        return _wrapper

                    setattr(mod, fn_name, _make_wrapper(fn))

            logger.debug("Patched hf_hub_download/snapshot_download for use_auth_token compat.")

        except Exception as e:
            logger.debug(f"huggingface_hub compat patch skipped: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Float32 numpy array (16 kHz mono).
            language: ISO 639-1 language code.

        Returns:
            Transcribed text string (lowercase).
        """
        if not self._available:
            return ""

        if self.engine == "wav2vec-sb":
            return self._transcribe_speechbrain(audio)
        else:
            return self._transcribe_transformers(audio, language)

    def _transcribe_transformers(self, audio: np.ndarray, language: str = "en") -> str:
        """Transcribe using HuggingFace transformers (standard + MMS)."""
        try:
            # MMS models: switch language adapter if needed
            if self._is_mms:
                iso3 = _ISO1_TO_ISO3.get(language, "eng")
                if self._current_lang != iso3:
                    try:
                        self._hf_processor.tokenizer.set_target_lang(iso3)
                        self._hf_model.load_adapter(iso3)
                        self._current_lang = iso3
                        logger.info(f"MMS language adapter switched to '{iso3}'.")
                    except Exception as e:
                        logger.warning(f"MMS adapter switch to '{iso3}' failed: {e}")

            input_values = self._hf_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_values.to(self.device)

            with torch.no_grad():
                logits = self._hf_model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            text = self._hf_processor.decode(predicted_ids[0])

            # wav2vec2 models output UPPERCASE — normalize to lowercase
            return text.strip().lower()

        except Exception as e:
            logger.error(f"Wav2Vec2 (transformers) transcription error: {e}")
            return ""

    def _transcribe_speechbrain(self, audio: np.ndarray) -> str:
        """Transcribe using SpeechBrain."""
        try:
            waveform = torch.from_numpy(audio).unsqueeze(0).float()
            wav_lens = torch.tensor([1.0])

            predicted_words, _ = self._sb_model.transcribe_batch(
                waveform, wav_lens
            )
            return predicted_words[0].strip()

        except Exception as e:
            logger.error(f"Wav2Vec2 (SpeechBrain) transcription error: {e}")
            return ""
