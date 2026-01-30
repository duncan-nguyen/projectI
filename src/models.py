from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Entity(BaseModel):
    id: str = Field(..., description="Unique identifier for the entity")
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: Optional[float] = None

class Relation(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    evidence: Optional[str] = None

class ProcessedDocument(BaseModel):
    original_text: str
    cleaned_text: str
    tokens: List[str]
    entities: List[Entity] = []
    relations: List[Relation] = []

class KnowledgeTriple(BaseModel):
    subject: str
    predicate: str
    object: str
