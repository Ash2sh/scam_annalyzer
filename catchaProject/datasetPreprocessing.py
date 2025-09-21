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

    text = re.sub(r"SPEAKER_\d+", "[SPEAKER]", text)

    # Телефоны
    text = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", "[PHONE]", text)

    # Email
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)

    # Адреса (очень простое правило: слова с "street", "st.", "ул.", "проспект", "avenue")
    text = re.sub(r"\b(?:street|st\.|ул\.|проспект|avenue|ave\.)\b.*?(?=,|\.|$)", "[ADDRESS]", text, flags=re.IGNORECASE)

    # Все числа (карты, паспорта и пр.)
    text = re.sub(r' \d[\d,.]*', '[NUMBER]', text)

    # Имена (очень базово: слова с заглавной буквы, кроме начала предложения и служебных слов)
    text = kp.replace_keywords(text)

    return text

def main():
    data = pd.read_csv("data/dataset/multi-agent_conversation_all.csv")
    dialogue = data["dialogue"].values
    for i, text in enumerate(dialogue):
        text = anonymize_text(text)
        dialogue[i] = text

    pd.DataFrame(data).to_csv("test.csv", index=False)
