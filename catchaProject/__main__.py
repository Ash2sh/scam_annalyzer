from catchaProject.audioProcessing import Audio, AudioProcessor
from catchaProject.checkScam import ScamAnalyzer

if __name__ == '__main__':
    audio = Audio("data/audio/input.mp3")
    processor = AudioProcessor()
    processor(audio)
    audio.save_to_json("data/transcript.json")
    
    analyzer = ScamAnalyzer()
    try:
        analyzer.run_analysis("transcript.json")
    except Exception as e:
        print(f"Analysis failed: {e}")
