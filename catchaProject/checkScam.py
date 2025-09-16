import json
import re

import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

from catchaProject.config import API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


class ScamAnalyzer:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=50, random_state=42)

    def run_analysis(self, transcript_path: str):
        """ Основной метод для анализа транскрипта и определения вероятности мошенничества. """

        diarized_text = self.load_transcript(transcript_path)
        prompt = self.build_prompt(diarized_text)
        llm_response = self.query_llm(prompt)
        if not llm_response:
            raise RuntimeError("Failed to get response from LLM.")

        speaker = self.extract_probabilities(llm_response)
        X, y_labels = self.prepare_features(diarized_text, speaker)
        self.fit_classifier(X, y_labels)
        return self.predict_probabilities(diarized_text, X)

    def load_transcript(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_prompt(self, diarized_text, max_segments=20) -> str:
        """ Формирует prompt для LLM на основе диаризованного транскрипта. """

        transcript_str = "\n".join(
            f"{seg['speaker']}: {seg['text']}" for seg in diarized_text[:max_segments]
        )
        prompt = (
            "Ты аналитик колл-центра. Оцени вероятность мошенничества для каждого спикера.\n"
            "Дай ответ в формате: SPEAKER: X% мошенничество.\n"
            "Транскрипт:\n"
            f"{transcript_str}\n"
        )
        return prompt

    def query_llm(self, prompt):
        """ Отправляет запрос к LLM и возвращает ответ. """

        headers = {
            "Authorization": f"Bearer {API_KEY}",
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
        content: str = llm_response["choices"][0]["message"]["content"]
        """ Извлекает вероятности мошенничества для каждого спикера из ответа LLM. """

        speaker = ("UNKNOWN", 0)
        for i, part in enumerate(re.findall(r"(\d+)%", content)):
            prob = int(part)
            if prob > speaker[1]:
                speaker = (f"SPEAKER_{i:02d}", prob)
        return speaker

    def prepare_features(
        self, diarized_text: list[dict], speaker: tuple[str, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Формирует признаки и метки для классификатора."""

        X = []
        y = []

        for seg in diarized_text:
            audio_feats = seg.get("audio_feats", [0, 0, 0])
            X.append([speaker[1]] + audio_feats)
            y.append(1 if seg["speaker"] == speaker[0] else 0)

        return np.array(X), np.array(y)

    def fit_classifier(self, X, y_labels):
        self.clf.fit(X, y_labels)

    def predict_probabilities(self, diarized_text, X):
        """ Предсказывает вероятность мошенничества для каждого спикера и выводит результаты. """

        result = {}
        for i, seg in enumerate(diarized_text):
            final_prob = self.clf.predict_proba([X[i]])[0, 1]
            print(
                f"{seg['speaker']} → {final_prob*100:.1f}% мошенничество, текст: '{seg['text']}'"
            )
            if seg["speaker"] not in result:
                result[seg["speaker"]] = {"sum": 0, "max": 0, "count": 0}
            speaker_data = result[seg["speaker"]]
            speaker_data["sum"] += final_prob
            speaker_data["max"] = max(speaker_data["max"], final_prob)
            speaker_data["count"] += 1

        for speaker, data in result.items():
            avg = data["sum"] / data["count"]
            print(
                f"{speaker}: средняя вероятность = {avg*100:.1f}%, макс = {data['max']*100:.1f}%"
            )

        return result