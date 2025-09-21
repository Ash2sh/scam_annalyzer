import json
from typing import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional
from transformers import BertForSequenceClassification, BertTokenizer

from catchaProject.config import BERT_MODEL, BERT_TOKENIZER


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str
    audio_feats: list[float]

    def __str__(self):
        return f"{self.speaker}: {self.text}"


class Transcript:
    def __init__(self, segments: list[Segment] | list[dict]):
        self.segments = segments
        self._changed = False

    @staticmethod
    def json_load(filePath: str) -> "Transcript":
        with open(filePath, "r", encoding="utf-8") as f:
            return Transcript(json.load(f))

    def normalize(self) -> None:
        stack: list[Segment] = []
        for seg in self:
            if stack:
                lastSegment = stack[-1]
                if lastSegment.speaker == seg.speaker:
                    lastSegment.text += " " + seg.text
                    lastSegment.end = seg.end
                    continue
            stack.append(seg)
        self.segments = stack
        self._changed = True

    def combine_replics(self) -> list[str]:
        texts = []
        second = False
        for seg in self:
            if second:
                texts[-1] += " " + str(seg)
                second = False
            else:
                texts.append(str(seg))
                second = True
        return texts

    def __len__(self):
        return len(self.segments)

    def __iter__(self) -> Iterator[Segment]:
        for segment in self.segments:
            if self._changed:
                yield segment
                continue
            yield Segment(
                segment["speaker"],
                segment["start"],
                segment["end"],
                segment["text"],
                segment["audio_feats"],
            )


class ScamAnalyzer:
    def run_analysis(self, transcript_path: str):
        """Основной метод для анализа транскрипта и определения вероятности мошенничества."""

        transcript: Transcript = Transcript.json_load(transcript_path)
        transcript.normalize()

        texts = transcript.combine_replics()
        return self.predict_probabilities(texts)

    def predict_probabilities(self, texts: list[str]):
        """Предсказывает вероятность мошенничества и выводит результаты."""

        model = BertForSequenceClassification.from_pretrained(BERT_MODEL)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER)

        sdsd = []
        for i in np.array_split(texts, round(len(texts)/5)):
            inputs = tokenizer(
                list(i),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            probs = functional.softmax(logits, dim=1).numpy()[0]
            pred_class = probs.argmax()

            print("Predicted class:", pred_class)
            print("Probabilities:", probs)
            sdsd.append(probs[1])

        print(sum(sdsd)/len(sdsd))
