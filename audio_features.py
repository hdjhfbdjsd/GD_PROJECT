import librosa
import numpy as np
import os

class AudioFeatures:
    def __init__(self):
        pass

    def extract_features(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        signal, sr = librosa.load(audio_path, sr=None)

        features = {
            "duration": librosa.get_duration(y=signal, sr=sr),
            "sample_rate": sr,
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(signal))),
            "energy": float(np.mean(signal ** 2)),
            "rms": float(np.mean(librosa.feature.rms(y=signal))),
            "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))),
            "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))),
            "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))),
            "mfcc": np.mean(librosa.feature.mfcc(y=signal, sr=sr), axis=1).tolist()
        }

        return features
