"""
deepfilternet_model.py — Wrapper for the DeepFilterNet3 noise suppression model.

Loads the pre-trained DeepFilterNet3 model from a local directory and provides
a simple `enhance()` interface for real-time noise suppression.

Key detail: DeepFilterNet operates at 48 kHz internally, so audio must be
resampled from 16 kHz before enhancement and back to 16 kHz afterward.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class DeepFilterNetModel:
    """
    Wraps DeepFilterNet3 for real-time noise suppression.

    The model is loaded once from a local checkpoint directory and reused
    across all requests. Audio is resampled 16 kHz ↔ 48 kHz transparently.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        """
        Load the DeepFilterNet3 model from a local directory.

        Args:
            model_dir: Path to the DeepFilterNet3 model directory
                       (must contain config.ini and checkpoints/).
            device: "cuda" or "cpu".
        """
        self.device = device
        self.model = None
        self.df_state = None
        self._available = False

        try:
            from df.enhance import init_df, enhance
            self._enhance_fn = enhance

            # Load model from local directory
            logger.info(f"Loading DeepFilterNet3 from {model_dir} on {device}...")
            self.model, self.df_state, _ = init_df(
                model_base_dir=model_dir,
                post_filter=True,
            )
            self._available = True
            logger.info("DeepFilterNet3 loaded successfully.")

        except Exception as e:
            logger.warning(f"DeepFilterNet3 could not be loaded: {e}")
            logger.warning("Noise suppression will be unavailable.")

    @property
    def available(self) -> bool:
        """Whether the model was loaded successfully."""
        return self._available

    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Apply noise suppression to an audio chunk.

        Args:
            audio: Float32 numpy array of audio samples (at `sample_rate` Hz).
            sample_rate: Input/output sample rate (default 16 kHz).

        Returns:
            Enhanced audio as float32 numpy array at the same sample rate.
        """
        if not self._available or self.model is None:
            return audio

        try:
            import torch
            from scipy.signal import resample_poly

            # ── Resample 16 kHz → 48 kHz for DeepFilterNet ──────────────
            if sample_rate != 48000:
                ratio_up = 48000 // sample_rate  # 3 for 16kHz→48kHz
                audio_48k = resample_poly(audio, up=ratio_up, down=1).astype(np.float32)
            else:
                audio_48k = audio

            # ── Enhance ──────────────────────────────────────────────────
            audio_tensor = torch.from_numpy(audio_48k).unsqueeze(0)
            enhanced = self._enhance_fn(
                self.model, self.df_state, audio_tensor
            )
            enhanced_np = enhanced.squeeze().numpy()

            # ── Resample 48 kHz → 16 kHz ────────────────────────────────
            if sample_rate != 48000:
                enhanced_np = resample_poly(
                    enhanced_np, up=1, down=ratio_up
                ).astype(np.float32)

            return enhanced_np

        except Exception as e:
            logger.error(f"DeepFilterNet enhance error: {e}")
            return audio
