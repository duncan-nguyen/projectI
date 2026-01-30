from .models import Entity, Relation

class RuleBasedRE:
    def __init__(self, char_threshold: int = 100):
        self.char_threshold = char_threshold
        self.rules = {
            "LIVED_AT": {
                "keywords": ["trú tại", "địa chỉ", "ở tại", "thường trú"],
                "source_type": ["PATIENT_ID", "NAME", "GENDER"], # Often Patient ID or Name is the subject
                "target_type": ["LOCATION"]
            },
            "HAS_SYMPTOM": {
                "keywords": ["sốt", "ho", "biểu hiện", "triệu chứng", "đau", "mệt"],
                "source_type": ["PATIENT_ID", "NAME"],
                "target_type": ["SYMPTOM_AND_DISEASE"]
            },
            "VISITED": {
                "keywords": ["đi đến", "tới", "di chuyển", "có mặt", "ngồi"],
                "source_type": ["PATIENT_ID", "NAME"],
                "target_type": ["LOCATION"]
            }
        }

    def extract(self, text: str, entities: list[Entity]) -> list[Relation]:
        relations = []
        sorted_entities = sorted(entities, key=lambda x: x.start_char)
        
        for i in range(len(sorted_entities)):
            for j in range(len(sorted_entities)):
                if i == j:
                    continue
                
                subj = sorted_entities[i]
                obj = sorted_entities[j]
                
                dist = abs(subj.end_char - obj.start_char)
                if dist > self.char_threshold:
                    continue
                    
                start = min(subj.end_char, obj.end_char)
                end = max(subj.start_char, obj.start_char)
                context = text[start:end].lower() if start < end else ""
                
                for rel_type, rule in self.rules.items():
                    if (subj.label in rule["source_type"] and 
                        obj.label in rule["target_type"]):
                        
                        for kw in rule["keywords"]:
                            if kw in context:
                                relations.append(Relation(
                                    source_id=subj.id,
                                    target_id=obj.id,
                                    relation_type=rel_type,
                                    evidence=kw
                                ))
                                break 
        return relations
