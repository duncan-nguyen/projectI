import re
import unicodedata
from typing import List

class TextPreprocessor:
    def __init__(self, segmenter_func=None):
        self.segmenter = segmenter_func

    def clean_text(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('\u200b', '')
        text = unicodedata.normalize('NFC', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        if self.segmenter:
            try:
                return self.segmenter(text)
            except Exception as e:
                print(f"Segmentation error: {e}")
                return text.split()
        return text.split()