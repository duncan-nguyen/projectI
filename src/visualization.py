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
    def generate_knowledge_graph(
        entities: List[Entity], relations: List[Relation]
    ) -> str:
        mermaid_code = ["graph TD"]

        for ent in entities:
            safe_id = ent.id.replace("-", "_")
            safe_label = ent.text.replace('"', "").replace("'", "")
            mermaid_code.append(
                f'    {safe_id}("{safe_label}<br/><small>{ent.label}</small>")'
            )

        for rel in relations:
            src_safe = rel.source_id.replace("-", "_")
            tgt_safe = rel.target_id.replace("-", "_")
            mermaid_code.append(
                f'    {src_safe} -- "{rel.relation_type}" --> {tgt_safe}'
            )

        return "\n".join(mermaid_code)

    @staticmethod
    def create_networkx_graph(entities: List[Entity], relations: List[Relation]):
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        # Define colors for different entity types
        color_map = {
            "PATIENT": "#FF9999",  # Red
            "LOCATION": "#99CCFF", # Blue
            "SYMPTOM": "#FFFF99",  # Yellow
            "DATE": "#99FF99",     # Green
            "JOB": "#FFCC99",      # Orange
            "ORGANIZATION": "#CC99FF", # Purple
            "default": "#E0E0E0"   # Grey
        }
        
        node_colors = []
        labels = {}
        
        for ent in entities:
            G.add_node(ent.id, label=ent.label, text=ent.text)
            labels[ent.id] = ent.text
            
            # Determine color
            c = color_map["default"]
            for key, val in color_map.items():
                if key in ent.label:
                    c = val
                    break
            node_colors.append(c)
            
        edge_labels = {}
        for rel in relations:
            G.add_edge(rel.source_id, rel.target_id, relation=rel.relation_type)
            edge_labels[(rel.source_id, rel.target_id)] = rel.relation_type
            
        # Draw
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42, k=1.5)  # k regulates distance
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, arrowsize=20, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family="sans-serif", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis("off")
        
        return fig
