import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import functional
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from catchaProject.config import (
    BERT_MODEL,
    BERT_TOKENIZER,
    DATASET_TEST,
    DATASET_TRAIN,
    TORCH_DEVICE,
)


class ScamDataset(Dataset):
    def __init__(self, dialogue, labels, tokenizer: BertTokenizer, max_len=128):
        self.dialogue = dialogue
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dialogue)

    def __getitem__(self, idx):
        text = str(self.dialogue.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


def train():
    train_df = DATASET_TRAIN.sample(frac=1, random_state=42)
    val_df = DATASET_TEST.sample(frac=1, random_state=42)[:320]

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased", do_lower_case=True
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=2
    ).to(TORCH_DEVICE)

    train_dataset = ScamDataset(train_df["dialogue"], train_df["label"], tokenizer, 256)
    val_dataset = ScamDataset(val_df["dialogue"], val_df["label"], tokenizer, 256)

    training_args = TrainingArguments(
        output_dir="./fraud-bert",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=0.7,
        weight_decay=0.01,
        logging_steps=50,
        learning_rate=2e-5,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(BERT_MODEL)
    tokenizer.save_pretrained(BERT_TOKENIZER)


def test():
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER)
    text = """SPEAKER_00: Добрый день! Я хочу обновить данные по своей карте в вашем сервисе.
SPEAKER_01: Здравствуйте! Отлично, мы как раз делаем акцию: если вы прямо сейчас подтвердите реквизиты, получите бонус 500 рублей.
SPEAKER_00: Какие реквизиты нужно подтвердить?
SPEAKER_01: Только номер карты и CVV. Это для активации бонуса.
SPEAKER_00: А где я могу проверить, что это безопасно?
SPEAKER_01: Не волнуйтесь, это внутренняя проверка, все данные защищены. После этого бонус сразу зачисляется.
SPEAKER_00: Спасибо, я сначала уточню у банка.
"""
    texts = list(map(" ".join, np.array_split(text.split("\n"), 4)))

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = functional.softmax(logits, dim=1).numpy()[0]
    pred_class = probs.argmax()

    print("Predicted class:", pred_class)
    print("Probabilities:", probs)
