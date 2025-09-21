from catchaProject.audioProcessing import Audio, AudioProcessor
from catchaProject.checkScam import ScamAnalyzer

# from catchaProject.bert import train, test
# from catchaProject.datasetPreprocessing import main


if __name__ == "__main__":
    audio = Audio("data/audio/videoplayback.m4a")
    processor = AudioProcessor()
    processor(audio)
    audio.save_to_json("data/transcript.json")

    analyzer = ScamAnalyzer()
    try:
        analyzer.run_analysis("transcript.json")
    except Exception as e:
        print(f"Analysis failed: {e}")
    # train()
    # test()
    # main()
