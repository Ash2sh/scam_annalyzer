import json

import librosa
import numpy as np
import torch
import whisperx
from whisperx.asr import FasterWhisperPipeline
from whisperx.diarize import DiarizationPipeline
from whisperx.transcribe import TranscriptionResult
from whisperx.types import SingleSegment

from catchaProject.config import (
    DIARIZATION_MODEL_NAME,
    DOWNLOAD_ROOT,
    HF_TOKEN,
    TORCH_DEVICE,
    WHISPER_MODEL_NAME,
)

BATCH_SIZE = 16
SAMPLERATE = 16000


def load_whisper_model():
    """
    Загрузка WhisperX модели
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return whisperx.load_model(
        WHISPER_MODEL_NAME, device=TORCH_DEVICE.type, download_root=DOWNLOAD_ROOT
    )


class Audio:
    def __init__(self, path: str):
        self.path = path
        self.sr = SAMPLERATE
        self._y: np.ndarray = whisperx.load_audio(self.path, self.sr)
        self.dialogue = []

    @property
    def y(self) -> np.ndarray:
        return self._y

    def save_to_json(self, filename="data/transcript.json"):
        """
        Сохраняем результат в JSON
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.dialogue, f, ensure_ascii=False, indent=2)


class AudioProcessor:
    """
    Класс для обработки аудио
    """

    def __init__(self):
        self.model: FasterWhisperPipeline = load_whisper_model()
        self.diarize_pipeline = DiarizationPipeline(
            DIARIZATION_MODEL_NAME,
            use_auth_token=HF_TOKEN,
            device=TORCH_DEVICE,
        )

    def __call__(self, audio: Audio):
        """
        Запуск обработки аудио

        Args:
            audio (Audio): Объект аудио

        Returns:
            Audio: Обработанный объект аудио
        """
        y = audio.y
        transcription = self.transcribe(y)
        diarization = self.diarize(y)
        result_aligned: TranscriptionResult = whisperx.assign_word_speakers(
            diarization, transcription
        )

        for seg in result_aligned["segments"]:
            segment = self.read_segment(seg, y)
            audio.dialogue.append(segment)

        return audio

    def transcribe(self, y: np.ndarray) -> TranscriptionResult:
        """
        Транскрипция аудио
        """
        transcription: TranscriptionResult = self.model.transcribe(
            y, batch_size=BATCH_SIZE
        )
        model_a, metadata = whisperx.load_align_model(
            language_code=transcription["language"], device=TORCH_DEVICE
        )
        transcription = whisperx.align(
            transcription["segments"],
            model_a,
            metadata,
            y,
            TORCH_DEVICE,
            return_char_alignments=False,
        )
        return transcription

    def diarize(self, y: np.ndarray):
        """
        Диаризация (разделение на спикеров)
        """
        return self.diarize_pipeline(y, num_speakers=2)

    def read_segment(self, segment: SingleSegment, y: np.ndarray) -> dict:
        """
        Чтение сегмента аудио
        """
        speaker = segment.get("speaker", "unknown")
        start_time = segment["start"]
        end_time = segment["end"]

        start_sample = int(start_time * SAMPLERATE)
        end_sample = int(end_time * SAMPLERATE)

        segment_audio = y[start_sample:end_sample]
        audio_feats = self.extract_audio_features(
            segment_audio, SAMPLERATE, segment["text"]
        )

        entry = {
            "speaker": speaker,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "text": segment["text"].strip(),
            "audio_feats": audio_feats,
        }
        print(f"{start_time:.1f}s - {end_time:.1f}s: {speaker}: {segment['text']}")

        return entry

    def extract_audio_features(self, audio, sample_rate, transcript):
        """
        Извлечение аудио признаков
        """
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        pitch_mean = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0

        energy = float(np.mean(audio**2))

        duration = len(audio) / sample_rate
        speech_rate = len(transcript.split()) / duration if duration > 0 else 0

        return [pitch_mean, energy, speech_rate]
