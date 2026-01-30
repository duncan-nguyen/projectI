from .models import ProcessedDocument
from .ner import BaseNER, GLiNERModel, StandardNER
from .preprocessing import TextPreprocessor
from .re_module import RuleBasedRE
from .visualization import Visualizer


class InformationExtractionPipeline:
    def __init__(
        self, ner_method: str = "gliner", model_path: str = None, device: str = "cpu"
    ):
        self.preprocessor = TextPreprocessor()

        if ner_method == "standard":
            path = model_path if model_path else "nqdhocai/vihealthbert-ner-v1"
            self.ner_model: BaseNER = StandardNER(model_path=path, device=device)
        elif ner_method == "gliner":
            path = model_path if model_path else "nqdhocai/med-gliner-v1"
            self.ner_model: BaseNER = GLiNERModel(model_path=path, device=device)
        else:
            raise ValueError("Invalid NER method. Choose 'standard' or 'gliner'")

        self.re_module = RuleBasedRE()

    def process(self, text: str) -> ProcessedDocument:
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.tokenize(cleaned_text)

        entities = self.ner_model.predict(cleaned_text, tokens)

        relations = self.re_module.extract(cleaned_text, entities)

        return ProcessedDocument(
            original_text=text,
            cleaned_text=cleaned_text,
            tokens=tokens,
            entities=entities,
            relations=relations,
        )

    def visualize(self, doc: ProcessedDocument):
        print("=== Entities ===")
        Visualizer.display_entities(doc.entities)

        print("\n=== Knowledge Graph (Mermaid) ===")
        print(Visualizer.generate_knowledge_graph(doc.entities, doc.relations))
