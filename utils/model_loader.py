from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import certifi
import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import GLOBAL_LOGGER as log


class ModelLoader:
    """
    Unified model loader for embeddings and LLMs.

    Notes:
    - Embeddings: supports offline-first loading with optional fallback to download.
      Also supports GPU acceleration via sentence-transformers device="cuda".
    - LLM (Ollama): supports optional generation controls (num_predict, temperature, num_ctx, etc.)
      via config/models.yaml keys.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f) or {}

            models: Dict[str, Any] = self.cfg.get("models", {}) or {}

            # Model names
            self.llm_name: str = models.get("llm_name", "llama3")
            self.embedding_model: str = models.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            )

            # HuggingFace cache options
            self.hf_cache_dir: str = models.get("hf_cache_dir", "data/cache/hf")
            self.local_files_only: bool = bool(models.get("local_files_only", False))
            self.allow_download_if_missing: bool = bool(models.get("allow_download_if_missing", False))

            # Embedding acceleration / quality controls
            # embedding_device: "cuda" | "cpu" | "auto"
            self.embedding_device: str = str(models.get("embedding_device", "auto")).lower()
            self.embedding_batch_size: int = int(models.get("embedding_batch_size", 128))
            self.normalize_embeddings: bool = bool(models.get("normalize_embeddings", True))

            # Ollama generation controls (optional, but recommended for speed)
            self.temperature: Optional[float] = models.get("temperature", None)
            self.num_predict: Optional[int] = models.get("num_predict", None)  # cap output tokens
            self.num_ctx: Optional[int] = models.get("num_ctx", None)          # context window
            self.top_p: Optional[float] = models.get("top_p", None)
            self.top_k: Optional[int] = models.get("top_k", None)

            log.info(
                "ModelLoader initialized",
                llm=self.llm_name,
                embed=self.embedding_model,
                hf_cache_dir=self.hf_cache_dir,
                local_files_only=self.local_files_only,
                allow_download_if_missing=self.allow_download_if_missing,
                embedding_device=self.embedding_device,
                embedding_batch_size=self.embedding_batch_size,
                normalize_embeddings=self.normalize_embeddings,
                temperature=self.temperature,
                num_predict=self.num_predict,
                num_ctx=self.num_ctx,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        except Exception as e:
            log.error("Failed to load model config", error=str(e))
            raise DocumentPortalException("ModelLoader initialization error", e)

    def load_llm(self):
        """
        Load Ollama LLM with optional generation controls.
        Recommended config for speed (CPU):
          models:
            num_predict: 800
            temperature: 0.2
            num_ctx: 2048
        """
        try:
            kwargs: Dict[str, Any] = {"model": self.llm_name}

            # Only pass options if user provided them in config
            if self.temperature is not None:
                kwargs["temperature"] = float(self.temperature)
            if self.num_predict is not None:
                kwargs["num_predict"] = int(self.num_predict)
            if self.num_ctx is not None:
                kwargs["num_ctx"] = int(self.num_ctx)
            if self.top_p is not None:
                kwargs["top_p"] = float(self.top_p)
            if self.top_k is not None:
                kwargs["top_k"] = int(self.top_k)

            llm = Ollama(**kwargs)
            log.info(
                "LLM loaded successfully",
                model=self.llm_name,
                **{k: v for k, v in kwargs.items() if k != "model"},
            )
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error", e)

    def _pick_embedding_device(self) -> str:
        """
        Decide which device to use for sentence-transformers embeddings.
        - If embedding_device="cpu": always CPU
        - If embedding_device="cuda": require CUDA, else fall back to CPU with warning
        - If embedding_device="auto": use CUDA if available else CPU
        """
        device = self.embedding_device

        try:
            import torch  # local import to avoid forcing torch for non-embedding usage
            cuda_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        except Exception:
            cuda_ok = False
            gpu_name = None

        if device == "cpu":
            log.info("Embeddings device selected", device="cpu")
            return "cpu"

        if device == "cuda":
            if cuda_ok:
                log.info("Embeddings device selected", device="cuda", gpu=gpu_name)
                return "cuda"
            log.warning("CUDA requested but not available; falling back to CPU")
            return "cpu"

        # auto
        if cuda_ok:
            log.info("Embeddings device selected", device="cuda", gpu=gpu_name)
            return "cuda"

        log.info("Embeddings device selected", device="cpu")
        return "cpu"

    def load_embeddings(self):
        """
        Loads HuggingFace embeddings using a local cache folder.

        Behavior:
        - If local_files_only=False: will download if missing.
        - If local_files_only=True:
            - will try offline load
            - if missing AND allow_download_if_missing=True: retry online download
            - else: raise a clear error telling you to cache or enable download

        GPU:
        - If models.embedding_device is "cuda" or "auto" and CUDA is available,
          sentence-transformers will run on GPU.
        """
        try:
            # SSL stability (Windows)
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            # reduce HF noise
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

            cache_dir = Path(self.hf_cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)

            device = self._pick_embedding_device()
            batch_size = max(1, int(self.embedding_batch_size))

            def _make(local_only: bool):
                # IMPORTANT:
                # - "device" controls GPU vs CPU for sentence-transformers
                # - "local_files_only" controls HF download behavior
                return HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    cache_folder=str(cache_dir),
                    model_kwargs={
                        "device": device,
                        "local_files_only": local_only,
                    },
                    encode_kwargs={
                        "batch_size": batch_size,
                        "normalize_embeddings": self.normalize_embeddings,
                    },
                )

            # 1) Try according to config
            try:
                emb = _make(self.local_files_only)
                log.info(
                    "Embeddings loaded successfully",
                    model=self.embedding_model,
                    cache_dir=str(cache_dir),
                    local_files_only=self.local_files_only,
                    device=device,
                    batch_size=batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                )
                return emb
            except Exception as e1:
                # 2) If offline requested but allowed to download, retry online
                if self.local_files_only and self.allow_download_if_missing:
                    log.warning(
                        "Embeddings not found in cache; retrying with download enabled",
                        model=self.embedding_model,
                        cache_dir=str(cache_dir),
                        device=device,
                    )
                    emb = _make(False)
                    log.info(
                        "Embeddings loaded successfully after download",
                        model=self.embedding_model,
                        cache_dir=str(cache_dir),
                        local_files_only=False,
                        device=device,
                        batch_size=batch_size,
                        normalize_embeddings=self.normalize_embeddings,
                    )
                    return emb

                # 3) Otherwise raise a clear error
                msg = (
                    "Embeddings model not found in local cache and downloads are disabled.\n"
                    f"- embedding_model: {self.embedding_model}\n"
                    f"- hf_cache_dir: {cache_dir}\n"
                    f"- local_files_only: {self.local_files_only}\n"
                    f"- allow_download_if_missing: {self.allow_download_if_missing}\n\n"
                    "Fix options:\n"
                    "1) Set models.local_files_only=false (allow downloads), OR\n"
                    "2) Keep local_files_only=true but pre-download/cache the model, OR\n"
                    "3) Set models.allow_download_if_missing=true to auto-download when missing.\n"
                )
                raise DocumentPortalException(msg, e1)

        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("Failed to load embeddings", error=str(e))
            raise DocumentPortalException("Embedding loading error", e)
