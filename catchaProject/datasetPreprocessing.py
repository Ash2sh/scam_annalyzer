import pandas as pd
import numpy as np
import time
import re
from flashtext import KeywordProcessor

kp = KeywordProcessor(case_sensitive=True)
with open("data/dataset/names.txt", "r") as f:
    names = f.readlines()
    for name in names:
        r = name.strip()
        if len(r) > 3:
            kp.add_keyword(r, "[NAME]")


def anonymize_text(text: str) -> str:
    """
    Заменяет имена, телефоны, email, цифры и адреса на специальные маркеры.
    """

    text = re.sub(r"SPEAKER_\d+|Suspect|Innocent", "[SPEAKER]", text)

    # Телефоны
    text = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", "[PHONE]", text)

    # Email
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)

    # Адреса (очень простое правило: слова с "street", "st.", "ул.", "проспект", "avenue")
    text = re.sub(r"\b(?:street|st\.|ул\.|проспект|avenue|ave\.)\b.*?(?=,|\.|$)", "[ADDRESS]", text, flags=re.IGNORECASE)

    # Все числа (карты, паспорта и пр.)
    text = re.sub(r' \d[\d,.]*', '[NUMBER]', text)

    # Имена
    text = kp.replace_keywords(text)

    return text

def normalize(data):
    final = {"dialogue": [], "label": []}
    for label, content in data.items():
        if label == "dialogue":
            for i in content:
                text = anonymize_text(i)
                final["dialogue"].append(text)
        elif label == "label":
            for i in content:
                final["label"].append(i)

    return pd.DataFrame(final)

def main():
    data1 = pd.read_csv("data/dataset/multi-agent_conversation_all.csv")
    data2 = pd.read_csv("data/dataset/single-agent-scam-dialogue_all.csv")

    df1 = normalize(data1)
    df2 = normalize(data2)
    df_all = pd.concat([df1, df2], ignore_index=True)
    df_all.to_csv("data/dataset/test.csv", index=False)
