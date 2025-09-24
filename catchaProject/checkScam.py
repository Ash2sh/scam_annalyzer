import json
import re
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import requests
import torch
from torch.nn import functional
from transformers import BertForSequenceClassification, BertTokenizer

from catchaProject.config import (
    BERT_MODEL,
    BERT_TOKENIZER,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
)


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
    # def run(self, transcript_path: str):
    #     """Основной метод для анализа транскрипта и определения вероятности мошенничества."""

    #     transcript: Transcript = Transcript.json_load(transcript_path)
    #     transcript.normalize()

    #     texts = transcript.combine_replics()
    #     return self.predict_probabilities(texts)

    def run(self, transcript_path: str):
        """Основной метод для анализа транскрипта и определения вероятности мошенничества."""

        transcript = Transcript.json_load(transcript_path)
        transcript.normalize()

        prompt = self.build_prompt(transcript)
        llm_response = self.query_llm(prompt)
        if not llm_response:
            raise RuntimeError("Failed to get response from LLM.")

        speaker = self.extract_probabilities(llm_response)
        return speaker

    def build_prompt(self, transcript, max_segments=20) -> str:
        """Формирует prompt для LLM на основе диаризованного транскрипта."""

        segments = []
        for i, seg in enumerate(transcript):
            if i <= max_segments:
                segments.append(str(seg))
            else:
                break
        transcript_str = "\n".join(segments)
        prompt = (
            "Ты аналитик колл-центра. Оцени вероятность мошенничества для каждого спикера.\n"
            "Дай ответ в формате: SPEAKER: X% мошенничество.\n"
            "Транскрипт:\n"
            f"{transcript_str}\n"
        )
        return prompt

    def query_llm(self, prompt):
        """Отправляет запрос к LLM и возвращает ответ."""

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты эксперт по выявлению мошенников в звонках.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        tries = 0
        while tries < 4:
            try:
                response = requests.post(
                    LLM_BASE_URL, headers=headers, json=payload, timeout=10
                )
                response.raise_for_status()
                return response.json()
            except requests.Timeout:
                print("Request timed out. Retrying...")
            except requests.RequestException as e:
                print(f"An error occurred: {e}")
            tries += 1
        return None

    def extract_probabilities(self, llm_response) -> tuple[str, int]:
        """Извлекает вероятности мошенничества для каждого спикера из ответа LLM."""

        content: str = llm_response["choices"][0]["message"]["content"]
        speaker = ("UNKNOWN", 0)
        for i, part in enumerate(re.findall(r"(\d+)%", content)):
            prob = int(part)
            if prob > speaker[1]:
                speaker = (f"SPEAKER_{i:02d}", prob)
        return speaker

    def predict_probabilities(self, texts: list[str]):
        """Предсказывает вероятность мошенничества и выводит результаты."""

        model = BertForSequenceClassification.from_pretrained(BERT_MODEL)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER)

        sdsd = []
        for i in np.array_split(texts, round(len(texts) / 8)):
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

        print(sum(sdsd) / len(sdsd))
