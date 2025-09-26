import logging
import asyncio

from catchaProject.bot import main

# from catchaProject.audioProcessing import Audio, AudioProcessor
# from catchaProject.checkScam import ScamAnalyzer

# from catchaProject.bert import train, test
# from catchaProject.datasetPreprocessing import main


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/bot.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    asyncio.run(main())

    # audio = Audio("data/audio/videoplayback.m4a")
    # processor = AudioProcessor()
    # processor(audio)
    # audio.save_to_json("transcript.json")

    # analyzer = ScamAnalyzer()
    # try:
    #     analyzer.run_analysis("transcript.json")
    # except Exception as e:
    #     print(f"Analysis failed: {e}")
    # train()
    # test()
    # main()
