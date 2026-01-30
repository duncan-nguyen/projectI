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
        for idx, res in enumerate(results):
            label = res["entity_group"]
            entities.append(
                Entity(
                    id=f"ent_{idx}",
                    text=res["word"],
                    label=label,
                    start_char=res["start"],
                    end_char=res["end"],
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
