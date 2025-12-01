from transcriber import Transcriber
from audio_features import AudioFeatures
from report_generator import ReportGenerator
from voice_generator import VoiceGenerator


def main():
    audio_file = "audio.mp3"

    # Step 1: Transcription
    transcriber = Transcriber(model_name="base")
    transcript = transcriber.transcribe(audio_file)

    # Step 2: Audio Features Extraction
    audio_features = AudioFeatures()
    features = audio_features.extract_features(audio_file)

    # Step 3: Report Generation
    report_generator = ReportGenerator(output_path="output/report.json")
    report = report_generator.generate(transcript, features)

    print("Transcription and audio feature extraction complete.")
    print("Report saved at: output/report.json")

     # 5) Text-to-Speech (voice generation) using pyttsx3
    print("Generating TTS audio from transcript...")

    # Read full transcript from file
    with open("output/transcript.txt", "r", encoding="utf-8") as f:
        tts_text = f.read().strip()

    if not tts_text:
        print("Transcript is empty, nothing to synthesize.")
    else:
        voice_gen = VoiceGenerator()
        voice_id="HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0"
        tts_output_path = voice_gen.synthesize(
            text=tts_text,
            output_path="output/tts_from_text.wav"
        )
        print(f"TTS audio saved to {tts_output_path}.\n")

    print("All steps completed successfully.")

if __name__ == "__main__":
    main()
