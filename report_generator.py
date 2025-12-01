import os
import json

class ReportGenerator:
    def __init__(self, output_path="output/report.json"):
        self.output_path = output_path

    def generate(self, transcript, audio_features):
        report = {
            "transcript": transcript,
            "audio_features": audio_features
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        return report
