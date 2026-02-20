from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import certifi
import yaml
from langchain_community.llms import Ollama

# Prefer the new, non-deprecated HuggingFace embeddings integration if installed.
# Fall back to langchain_community if not.
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    _USING_LANGCHAIN_HF = True
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    _USING_LANGCHAIN_HF = False

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import GLOBAL_LOGGER as log


class ModelLoader:
    """
    Unified model loader for embeddings and LLMs.

    Notes:
    - Embeddings:
        - Offline-first loading with optional fallback download.
        - GPU acceleration via device="cuda" when torch reports CUDA available.
        - Uses langchain-huggingface if installed (recommended), otherwise falls back
          to the deprecated langchain_community HuggingFaceEmbeddings.
    - LLM (Ollama):
        - Supports optional generation controls (num_predict, temperature, num_ctx, top_p, top_k).
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
            self.embedding_device: str = str(models.get("embedding_device", "auto")).lower().strip()
            self.embedding_batch_size: int = int(models.get("embedding_batch_size", 128))
            self.normalize_embeddings: bool = bool(models.get("normalize_embeddings", True))

            # Ollama generation controls (optional)
            self.temperature: Optional[float] = models.get("temperature", None)
            self.num_predict: Optional[int] = models.get("num_predict", None)
            self.num_ctx: Optional[int] = models.get("num_ctx", None)
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
                embeddings_backend=("langchain_huggingface" if _USING_LANGCHAIN_HF else "langchain_community"),
            )
        except Exception as e:
            log.error("Failed to load model config", error=str(e))
            raise DocumentPortalException("ModelLoader initialization error", e)

    def load_llm(self):
        """
        Load Ollama LLM with optional generation controls.

        Example (speed-friendly):
          models:
            num_predict: 800
            temperature: 0.2
            num_ctx: 2048
        """
        try:
            kwargs: Dict[str, Any] = {"model": self.llm_name}

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

    @staticmethod
    def _torch_cuda_status() -> Dict[str, Any]:
        """
        Safely query torch CUDA status. Never raises.
        """
        try:
            import torch  # local import

            cuda_ok = bool(torch.cuda.is_available())
            cuda_ver = getattr(torch.version, "cuda", None)
            gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
            return {"cuda_ok": cuda_ok, "cuda_version": cuda_ver, "gpu_name": gpu_name}
        except Exception:
            return {"cuda_ok": False, "cuda_version": None, "gpu_name": None}

    def _pick_embedding_device(self) -> str:
        """
        Decide which device to use for sentence-transformers embeddings.

        Rules:
        - "cpu": always CPU
        - "cuda": use CUDA if available else CPU (warn)
        - "auto": CUDA if available else CPU
        """
        req = self.embedding_device

        status = self._torch_cuda_status()
        cuda_ok = status["cuda_ok"]
        gpu_name = status["gpu_name"]
        cuda_ver = status["cuda_version"]

        if req == "cpu":
            log.info("Embeddings device selected", device="cpu")
            return "cpu"

        if req == "cuda":
            if cuda_ok:
                log.info("Embeddings device selected", device="cuda", gpu=gpu_name, torch_cuda=cuda_ver)
                return "cuda"
            log.warning("CUDA requested but not available; falling back to CPU", torch_cuda=cuda_ver)
            return "cpu"

        # auto
        if cuda_ok:
            log.info("Embeddings device selected", device="cuda", gpu=gpu_name, torch_cuda=cuda_ver)
            return "cuda"

        log.info("Embeddings device selected", device="cpu", torch_cuda=cuda_ver)
        return "cpu"

    def load_embeddings(self):
        """
        Load HuggingFace embeddings using a local cache folder.

        Behavior:
        - If local_files_only=False: will download if missing.
        - If local_files_only=True:
            - will try offline load
            - if missing AND allow_download_if_missing=True: retry online download
            - else: raise a clear error explaining how to fix

        GPU:
        - If embedding_device is "cuda" or "auto" and CUDA is available,
          sentence-transformers will run on GPU.
        """
        try:
            # SSL stability (Windows)
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())

            # reduce HF noise + telemetry
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

            cache_dir = Path(self.hf_cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)

            device = self._pick_embedding_device()
            batch_size = max(1, int(self.embedding_batch_size))

            def _make(local_only: bool):
                """
                Construct the embedding wrapper.

                IMPORTANT:
                - Put `device` in model_kwargs so SentenceTransformer moves to GPU.
                - Use local_files_only to control download behavior.
                """
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

                # Optional: force one tiny embed to ensure the model is really initialized on the device.
                # This helps catch lazy-init edge cases early.
                try:
                    _ = emb.embed_query("gpu_check")
                except Exception:
                    # If a wrapper doesn't support embed_query at init time, ignore.
                    pass

                log.info(
                    "Embeddings loaded successfully",
                    model=self.embedding_model,
                    cache_dir=str(cache_dir),
                    local_files_only=self.local_files_only,
                    device=device,
                    batch_size=batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                    embeddings_backend=("langchain_huggingface" if _USING_LANGCHAIN_HF else "langchain_community"),
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

                    try:
                        _ = emb.embed_query("gpu_check")
                    except Exception:
                        pass

                    log.info(
                        "Embeddings loaded successfully after download",
                        model=self.embedding_model,
                        cache_dir=str(cache_dir),
                        local_files_only=False,
                        device=device,
                        batch_size=batch_size,
                        normalize_embeddings=self.normalize_embeddings,
                        embeddings_backend=("langchain_huggingface" if _USING_LANGCHAIN_HF else "langchain_community"),
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