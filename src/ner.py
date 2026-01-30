from abc import ABC, abstractmethod

from gliner import GLiNER
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from .models import Entity


class BaseNER(ABC):
    @abstractmethod
    def predict(self, text: str, tokens: list[str] = None) -> list[Entity]:
        pass


class StandardNER(BaseNER):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = 0 if device == "cuda" else -1
        self.label_list = [
            "O",
            "B-AGE",
            "I-AGE",
            "B-DATE",
            "I-DATE",
            "B-GENDER",
            "I-GENDER",
            "B-JOB",
            "I-JOB",
            "B-LOCATION",
            "I-LOCATION",
            "B-NAME",
            "I-NAME",
            "B-ORGANIZATION",
            "I-ORGANIZATION",
            "B-PATIENT_ID",
            "I-PATIENT_ID",
            "B-SYMPTOM_AND_DISEASE",
            "I-SYMPTOM_AND_DISEASE",
            "B-TRANSPORTATION",
            "I-TRANSPORTATION",
        ]
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.nlp = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
        )

    def predict(self, text: str, tokens: list[str] = None) -> list[Entity]:
        input_text = text
        if tokens:
            input_text = " ".join(tokens)

        results = self.nlp(input_text)
        entities = []

        # Track current position for manual offset calculation if needed
        current_pos = 0

        for idx, res in enumerate(results):
            label = res["entity_group"]
            word = res["word"]
            start = res.get("start")
            end = res.get("end")

            # Fix for slow tokenizers (like PhoBERT's) not returning offsets in pipeline
            if start is None or end is None:
                # Naive search for the word in the text starting from current_pos
                # Handle possible mismatch in spacing/underscores if necessary,
                # but direct find is a reasonable fallback
                clean_word = word.strip().replace("_", " ")
                # Also consider the input text might have underscores if it was pre-segmented

                # Try finding exact word
                found_idx = input_text.find(word, current_pos)
                if found_idx == -1:
                    # Try finding with spaces instead of underscores
                    found_idx = input_text.find(clean_word, current_pos)

                if found_idx != -1:
                    start = found_idx
                    end = found_idx + len(word)  # approximation
                    # Update current_pos to avoid finding same entity again
                    current_pos = end
                else:
                    start = 0
                    end = 0

            entities.append(
                Entity(
                    id=f"ent_{idx}",
                    text=word,
                    label=label,
                    start_char=start,
                    end_char=end,
                    confidence=res["score"],
                )
            )
        return entities


class GLiNERModel(BaseNER):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = GLiNER.from_pretrained(model_path)
        if device == "cuda":
            self.model.to("cuda")

        self.labels = [
            "SYMPTOM_AND_DISEASE",
            "PATIENT_ID",
            "NAME",
            "AGE",
            "GENDER",
            "JOB",
            "LOCATION",
            "ORGANIZATION",
            "TRANSPORTATION",
            "DATE",
        ]

    def predict(self, text: str, tokens: list[str] = None) -> list[Entity]:
        preds = self.model.predict_entities(text, self.labels)

        entities = []
        for idx, p in enumerate(preds):
            entities.append(
                Entity(
                    id=f"gli_{idx}",
                    text=p["text"],
                    label=p["label"],
                    start_char=p["start"],
                    end_char=p["end"],
                    confidence=p["score"],
                )
            )
        return entities
