# ===============================================================
# config_schema.py — Pydantic schema for DMPCHEF-Pipeline config
# ===============================================================

from pydantic import BaseModel
from typing import Optional


class PathsConfig(BaseModel):
    """Defines all file and directory paths used in the pipeline."""
    data_pdfs: str
    index_dir: str
    excel_path: str
    output_md: str
    output_docx: str


class RAGConfig(BaseModel):
    """Defines all RAG-related parameters."""
    chunk_size: int = 800
    chunk_overlap: int = 120
    retriever_top_k: int = 3


class ModelsConfig(BaseModel):
    """Defines models used in the pipeline."""
    llm_name: str
    embedding_model: str


class ExperimentConfig(BaseModel):
    """Root config schema."""
    root_dir: str
    paths: PathsConfig
    rag: RAGConfig
    models: ModelsConfig
