from typing import List
from .models import Entity, Relation

class Visualizer:
    @staticmethod
    def display_entities(entities: List[Entity]):
        print(f"{'Text':<20} | {'Label':<20} | {'Confidence':<10}")
        print("-" * 55)
        for ent in entities:
            conf = f"{ent.confidence:.2f}" if ent.confidence else "N/A"
            print(f"{ent.text:<20} | {ent.label:<20} | {conf:<10}")

    @staticmethod
    def generate_knowledge_graph(entities: List[Entity], relations: List[Relation]) -> str:
        mermaid_code = ["graph TD"]
        
        for ent in entities:
            safe_id = ent.id.replace("-", "_")
            safe_label = ent.text.replace('"', '').replace("'", "")
            mermaid_code.append(f'    {safe_id}("{safe_label}<br/><small>{ent.label}</small>")')
            
        for rel in relations:
            src_safe = rel.source_id.replace("-", "_")
            tgt_safe = rel.target_id.replace("-", "_")
            mermaid_code.append(f'    {src_safe} -- "{rel.relation_type}" --> {tgt_safe}')
            
        return "\n".join(mermaid_code)
