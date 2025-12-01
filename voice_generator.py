import os
import pyttsx3

class VoiceGenerator:
    def __init__(self, rate: int | None = None, volume: float | None = None, voice_id: str | None = None):
        """
        Simple TTS using pyttsx3 (offline, works with newer Python versions).
        rate: speech rate (words per minute), e.g. 150.
        volume: volume between 0.0 and 1.0.
        voice_id: optional specific voice id from available engine voices.
        """
        self.engine = pyttsx3.init()

        if rate is not None:
            self.engine.setProperty("rate", rate)

        if volume is not None:
            self.engine.setProperty("volume", volume)

        if voice_id is not None:
            self.engine.setProperty("voice", voice_id)

    def list_voices(self):
        """Print available voices (id and name)."""
        voices = self.engine.getProperty("voices")
        for v in voices:
            print(f"ID: {v.id} | Name: {v.name}")

    def synthesize(self, text: str, output_path: str = "output/tts_from_text.wav") -> str:
        """
        Generate speech from text and save to output_path.
        Note: Format and support can vary by OS; WAV works best on most systems.
        """
        if not text:
            raise ValueError("Text for TTS is empty.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save audio to file
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()

        return output_path
    

