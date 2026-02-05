# src/utils/model_loader.py
import os
from pathlib import Path

import certifi
import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import GLOBAL_LOGGER as log


class ModelLoader:
    """Unified model loader for embeddings and LLMs."""

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f) or {}

            models = self.cfg.get("models", {}) or {}
            self.llm_name = models.get("llm_name", "llama3")
            self.embedding_model = models.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            )

            self.hf_cache_dir = models.get("hf_cache_dir", "data/cache/hf")
            self.local_files_only = bool(models.get("local_files_only", False))

            log.info(
                "ModelLoader initialized",
                llm=self.llm_name,
                embed=self.embedding_model,
                hf_cache_dir=self.hf_cache_dir,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            log.error("Failed to load model config", error=str(e))
            raise DocumentPortalException("ModelLoader initialization error", e)

    def load_llm(self):
        try:
            llm = Ollama(model=self.llm_name)
            log.info("LLM loaded successfully", model=self.llm_name)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error", e)

    def load_embeddings(self):
        """
        Loads HuggingFace embeddings using a local cache folder so the model
        isn't re-downloaded every run.

        If local_files_only=True, it will NEVER hit the network.
        """
        try:
            # SSL stability (Windows)
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())

            # optional: reduce HF noise
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

            # cache dir
            cache_dir = Path(self.hf_cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)

            emb = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                cache_folder=str(cache_dir),
                model_kwargs={"local_files_only": self.local_files_only},
            )

            log.info(
                "Embeddings loaded successfully",
                model=self.embedding_model,
                cache_dir=str(cache_dir),
                local_files_only=self.local_files_only,
            )
            return emb

        except Exception as e:
            log.error("Failed to load embeddings", error=str(e))
            raise DocumentPortalException("Embedding loading error", e)
