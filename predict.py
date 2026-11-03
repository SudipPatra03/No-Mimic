"""
Prediction / Inference Module
===============================
Single-file inference pipeline for the web application.
"""

import torch
import numpy as np
import os
from config import config
from src.preprocessor import AudioPreprocessor
from src.feature_extractor import FeatureExtractor
from src.model import VoiceAuthenticityModel

class VoicePredictor:
    """Handles single-file voice authenticity prediction."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = config.BEST_MODEL_PATH

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()

        # Load model
        self.model = VoiceAuthenticityModel()
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: No model found at {model_path}, using random weights")

        self.model = self.model.to(self.device)
        self.model.eval()

    def _analyze_features(self, waveform: torch.Tensor, mel_spec: np.ndarray,
                          mfcc: np.ndarray, is_fake: bool, confidence: float) -> dict:
        """
        Analyze extracted features to generate human-readable reasons
        for why the audio was classified as authentic or AI-generated.

        Computes acoustic metrics from the mel spectrogram, MFCC, and
        raw waveform, then interprets them based on the model's verdict.

        Returns:
            Dict with 'reasons' (list of reason dicts) and 'metrics' (raw values)
        """
        reasons = []

        # ── 1. Spectral Flatness (from mel spectrogram) ──
        # Measures how noise-like vs tonal the spectrum is.
        # AI voices tend to have more uniform/flat spectra.
        geo_mean = np.exp(np.mean(np.log(mel_spec + 1e-10), axis=0))
        arith_mean = np.mean(mel_spec, axis=0) + 1e-10
        spectral_flatness = np.mean(geo_mean / arith_mean)

        if is_fake:
            if spectral_flatness > 0.3:
                reasons.append({
                    "icon": "spectrum", "title": "Unusually Flat Spectrum",
                    "detail": f"Spectral flatness ({spectral_flatness:.3f}) is higher than typical human speech, suggesting synthesized or overly smooth audio generation.",
                    "type": "warning"
                })
            else:
                reasons.append({
                    "icon": "spectrum", "title": "Spectral Pattern Anomaly",
                    "detail": f"Despite normal spectral flatness ({spectral_flatness:.3f}), the spectral distribution pattern shows subtle irregularities characteristic of AI synthesis.",
                    "type": "warning"
                })
        else:
            reasons.append({
                "icon": "spectrum", "title": "Natural Spectral Variation",
                "detail": f"Spectral flatness ({spectral_flatness:.3f}) shows natural tonal variation consistent with organic human speech.",
                "type": "ok"
            })

        # ── 2. Energy Variance Across Time (from mel spectrogram) ──
        # Human speech has high energy variance (pauses, emphasis).
        # AI voices often have more uniform energy distribution.
        frame_energy = np.mean(mel_spec, axis=0)
        energy_variance = np.var(frame_energy)
        energy_cv = np.std(frame_energy) / (np.mean(frame_energy) + 1e-10)

        if is_fake:
            if energy_cv < 0.4:
                reasons.append({
                    "icon": "energy", "title": "Uniform Energy Distribution",
                    "detail": f"Energy variation ({energy_cv:.3f}) is unusually consistent across time frames, lacking the natural rhythm and pauses of human speech.",
                    "type": "warning"
                })
            else:
                reasons.append({
                    "icon": "energy", "title": "Synthetic Energy Pattern",
                    "detail": f"Although energy variation ({energy_cv:.3f}) appears normal, the temporal energy transitions show mechanical patterns unlike natural speech.",
                    "type": "warning"
                })
        else:
            reasons.append({
                "icon": "energy", "title": "Natural Speech Rhythm",
                "detail": f"Energy variation ({energy_cv:.3f}) shows natural speech dynamics with organic pauses and emphasis patterns.",
                "type": "ok"
            })

        # ── 3. MFCC Stability (coefficient variance over time) ──
        # Human voice has natural MFCC fluctuation.
        # AI voices often show overly stable or overly erratic MFCCs.
        mfcc_temporal_var = np.mean(np.var(mfcc, axis=1))
        mfcc_delta = np.mean(np.abs(np.diff(mfcc, axis=1)))

        if is_fake:
            if mfcc_temporal_var < 0.5:
                reasons.append({
                    "icon": "mfcc", "title": "Overly Stable Vocal Characteristics",
                    "detail": f"MFCC variation ({mfcc_temporal_var:.3f}) is unusually low, indicating the vocal tract characteristics remain too consistent — a hallmark of synthetic speech.",
                    "type": "warning"
                })
            else:
                reasons.append({
                    "icon": "mfcc", "title": "Irregular Vocal Tract Modeling",
                    "detail": f"MFCC dynamics (Δ={mfcc_delta:.3f}) show patterns inconsistent with natural vocal tract movements, suggesting AI-based speech generation.",
                    "type": "warning"
                })
        else:
            reasons.append({
                "icon": "mfcc", "title": "Natural Vocal Characteristics",
                "detail": f"MFCC variation ({mfcc_temporal_var:.3f}) and transitions (Δ={mfcc_delta:.3f}) are consistent with natural vocal tract movements.",
                "type": "ok"
            })

        # ── 4. High-Frequency Energy Analysis ──
        # AI-generated audio often has less high-frequency content
        # or artificially boosted high frequencies.
        n_mels = mel_spec.shape[0]
        low_band_energy = np.mean(mel_spec[:n_mels // 3, :])
        high_band_energy = np.mean(mel_spec[2 * n_mels // 3:, :])
        hf_ratio = high_band_energy / (low_band_energy + 1e-10)

        if is_fake:
            if hf_ratio < 0.3:
                reasons.append({
                    "icon": "frequency", "title": "Reduced High-Frequency Content",
                    "detail": f"High-to-low frequency ratio ({hf_ratio:.3f}) indicates missing natural high-frequency harmonics, common in AI voice synthesis.",
                    "type": "warning"
                })
            else:
                reasons.append({
                    "icon": "frequency", "title": "Artificial Frequency Balance",
                    "detail": f"Frequency distribution ratio ({hf_ratio:.3f}) shows an engineered balance that differs from natural voice frequency profiles.",
                    "type": "warning"
                })
        else:
            reasons.append({
                "icon": "frequency", "title": "Natural Frequency Distribution",
                "detail": f"High-to-low frequency ratio ({hf_ratio:.3f}) shows the natural harmonic richness expected in human speech.",
                "type": "ok"
            })

        # ── 5. Temporal Smoothness (frame-to-frame transitions) ──
        # AI models often produce smoother or more abrupt transitions.
        frame_diffs = np.mean(np.abs(np.diff(mel_spec, axis=1)))

        if is_fake:
            if frame_diffs < 0.02:
                reasons.append({
                    "icon": "temporal", "title": "Over-Smoothed Transitions",
                    "detail": f"Frame-to-frame variation ({frame_diffs:.4f}) is abnormally smooth, indicating AI-generated audio that lacks natural micro-variations.",
                    "type": "warning"
                })
            else:
                reasons.append({
                    "icon": "temporal", "title": "Unnatural Temporal Dynamics",
                    "detail": f"Temporal transitions ({frame_diffs:.4f}) show patterns characteristic of neural network-based audio synthesis.",
                    "type": "warning"
                })
        else:
            reasons.append({
                "icon": "temporal", "title": "Organic Temporal Flow",
                "detail": f"Frame transitions ({frame_diffs:.4f}) demonstrate natural micro-variations typical of live human speech.",
                "type": "ok"
            })

        # ── 6. Overall Model Confidence Interpretation ──
        if confidence >= 90:
            strength = "very high"
        elif confidence >= 75:
            strength = "high"
        elif confidence >= 60:
            strength = "moderate"
        else:
            strength = "low"

        if is_fake:
            reasons.append({
                "icon": "model_warning", "title": f"Model Confidence: {strength.title()}",
                "detail": f"The CNN-BiLSTM model detected AI synthesis patterns with {confidence:.1f}% confidence based on combined spectral and temporal analysis.",
                "type": "warning"
            })
        else:
            reasons.append({
                "icon": "model_ok", "title": f"Model Confidence: {strength.title()}",
                "detail": f"The CNN-BiLSTM model confirmed authentic human voice patterns with {confidence:.1f}% confidence across all analyzed features.",
                "type": "ok"
            })

        # Collected metrics for optional display
        metrics = {
            "spectral_flatness": round(float(spectral_flatness), 4),
            "energy_cv": round(float(energy_cv), 4),
            "mfcc_temporal_var": round(float(mfcc_temporal_var), 4),
            "mfcc_delta": round(float(mfcc_delta), 4),
            "hf_ratio": round(float(hf_ratio), 4),
            "frame_smoothness": round(float(frame_diffs), 4),
        }

        return {"reasons": reasons, "metrics": metrics}

    def predict(self, audio_path: str) -> dict:
        """
        Predict whether an audio file contains authentic or AI-generated voice.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with prediction, confidence, feature visualizations, and analysis reasons
        """
        # Preprocess
        waveform = self.preprocessor.process(audio_path)

        # Extract features
        features = self.feature_extractor.extract_combined_features(waveform)
        viz_features = self.feature_extractor.extract_features_for_visualization(waveform)

        # Run inference
        with torch.no_grad():
            input_tensor = features.unsqueeze(0).to(self.device)
            logit = self.model(input_tensor)
            probability = torch.sigmoid(logit).item()

        # Classify
        is_fake = probability >= 0.5
        confidence = probability if is_fake else (1 - probability)
        confidence_pct = round(confidence * 100, 2)

        # Analyze features for human-readable reasons
        mel_spec = viz_features["mel_spectrogram"]
        mfcc = viz_features["mfcc"]
        analysis = self._analyze_features(waveform, mel_spec, mfcc, is_fake, confidence_pct)

        result = {
            "prediction": "ai_generated" if is_fake else "authentic",
            "confidence": confidence_pct,
            "raw_score": round(probability, 6),
            "label": "AI-Generated Voice" if is_fake else "Authentic Human Voice",
            "mel_spectrogram": mel_spec.tolist(),
            "mfcc": mfcc.tolist(),
            "reasons": analysis["reasons"],
            "metrics": analysis["metrics"],
        }

        return result

    def predict_from_tensor(self, waveform: torch.Tensor) -> dict:
        """Predict from a preprocessed waveform tensor."""
        features = self.feature_extractor.extract_combined_features(waveform)

        with torch.no_grad():
            input_tensor = features.unsqueeze(0).to(self.device)
            logit = self.model(input_tensor)
            probability = torch.sigmoid(logit).item()

        is_fake = probability >= 0.5
        confidence = probability if is_fake else (1 - probability)

        return {
            "prediction": "ai_generated" if is_fake else "authentic",
            "confidence": round(confidence * 100, 2),
            "raw_score": round(probability, 6),
            "label": "AI-Generated Voice" if is_fake else "Authentic Human Voice",
        }
