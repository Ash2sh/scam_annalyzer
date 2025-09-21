import os
import tomllib
import torch

import pandas as pd
from dotenv import load_dotenv

load_dotenv("config/.env")

with open("config/config.toml", "rb") as f:
    config = tomllib.load(f)

APP_NAME = config["app"]["name"]
DEBUG_MODE = config["app"]["debug"]

LOGGING_LEVEL = config["logging"]["level"]

TORCH_DEVICE = torch.device(config["torch"]["device"])

DOWNLOAD_ROOT = config["whisper"]["download_root"]
WHISPER_MODEL_NAME = config["whisper"]["model_name"]

DIARIZATION_MODEL_NAME = config["diarization"]["model_name"]

LLM_MODEL_NAME = config["llm"]["model_name"]
LLM_BASE_URL = config["llm"]["base_url"]

DATASET_TRAIN = pd.read_csv(config["dataset"]["train"])
DATASET_TEST = pd.read_csv(config["dataset"]["test"])

BERT_MODEL = config["bert"]["model"]
BERT_TOKENIZER = config["bert"]["tokenizer"]

API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")