import asyncio
from typing import Any, Dict, Optional

import librosa
import numpy as np
import structlog
import torch
import torchaudio

logger = structlog.get_logger()


class AudioProcessor:
    """Service for processing audio files and extracting features."""

    def __init__(self):
        self.sr = 22050  # Target sample rate for analysis
        self.n_mfcc = 13  # Number of MFCC features

    async def _load_audio_with_torchaudio(self, file_path: str) -> Optional[np.ndarray]:
        """
        Loads an audio file using torchaudio, which is self-contained and
        handles various backends robustly.
        """
        try:
            # waveform is a tensor, info contains metadata
            waveform, sample_rate = await asyncio.to_thread(torchaudio.load, file_path)

            # Resample if necessary to match our target sample rate
            if sample_rate != self.sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sr
                )
                waveform = resampler(waveform)

            # Convert to mono by averaging channels if it's stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Convert tensor to numpy array and ensure it's float32
            # Also, get rid of the channel dimension for librosa (shape from [1, n] to [n,])
            audio_np = waveform.squeeze().numpy().astype(np.float32)

            return audio_np

        except Exception as e:
            logger.error(
                "Error loading audio with torchaudio", error=str(e), file=file_path
            )
            return None

    async def extract_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract audio features from file."""
        # Use our new torchaudio loader
        y = await self._load_audio_with_torchaudio(file_path)

        if y is None:
            return None

        try:
            # Run the synchronous librosa functions in a thread
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(
                None, self._extract_features_sync, y, self.sr
            )
            logger.info("Audio features extracted successfully", file=file_path)
            return features

        except Exception as e:
            logger.error(
                "Error during librosa feature extraction", error=str(e), file=file_path
            )
            return None

    def _extract_features_sync(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Synchronous feature extraction using librosa."""
        # This function remains the same
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        rms = librosa.feature.rms(y=y).mean()
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).mean(axis=1)

        return {
            "tempo": float(tempo),
            "spectral_centroid": float(spectral_centroid),
            "rms_energy": float(rms),
            "chroma": [float(x) for x in chroma],
            "mfccs": [float(x) for x in mfccs],
        }
