# src/build_index.py
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from tqdm import tqdm

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.model_loader import ModelLoader
from logger.custom_logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


class ConfigManager:
    """
    Reads config/config.yaml and resolves relative paths using:
      1) config.root_dir (if present)
      2) repo_root (parent of /src)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        repo_root = Path(__file__).resolve().parents[1]  # <repo_root>/src/... -> <repo_root>
        raw_cfg = Path(config_path)
        cfg_path = raw_cfg if raw_cfg.is_absolute() else (repo_root / raw_cfg).resolve()

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.repo_root = repo_root
        self.cfg_path = cfg_path

        self.root_dir = self._resolve_root_dir(self.cfg.get("root_dir"))
        self.paths = self.cfg.get("paths", {}) or {}
        self.rag = self.cfg.get("rag", {}) or {}
        self.models = self.cfg.get("models", {}) or {}

        log.info(
            "Build-index config loaded",
            config=str(cfg_path),
            root_dir=str(self.root_dir),
        )

    def _resolve_root_dir(self, root_dir_value: Optional[str]) -> Path:
        if not root_dir_value:
            return self.repo_root

        p = Path(str(root_dir_value)).expanduser()
        if p.is_absolute():
            return p.resolve()

        # root_dir is relative to repo root
        return (self.repo_root / p).resolve()

    def resolve_path(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self.root_dir / p).resolve()

    def get_path(self, key: str) -> Path:
        val = self.paths.get(key)
        if not val:
            raise KeyError(f"Missing config.paths.{key} in YAML")
        return self.resolve_path(val)

    def get_rag_param(self, key: str):
        return self.rag.get(key)

    def get_models_param(self, key: str, default=None):
        return self.models.get(key, default)


def _load_one_pdf(pdf_path: Path) -> Tuple[list, Optional[Tuple[str, str, str]]]:
    """
    Load a single PDF robustly:
    - try PyMuPDFLoader first
    - fallback to PyPDFLoader
    Returns:
      (docs, None) on success
      ([], (pdf_path, pymupdf_error, pypdf_error)) on failure
    """
    try:
        return PyMuPDFLoader(str(pdf_path)).load(), None
    except Exception as e1:
        try:
            return PyPDFLoader(str(pdf_path)).load(), None
        except Exception as e2:
            return [], (str(pdf_path), str(e1), str(e2))


class IndexBuilder:
    """
    Completely separate module:
    PDFs -> chunks -> embeddings -> FAISS -> save_local(index_dir)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config = ConfigManager(config_path=config_path)

            self.data_pdfs = self.config.get_path("data_pdfs")
            self.index_dir = self.config.get_path("index_dir")
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Embeddings model comes from your existing ModelLoader
            self.model_loader = ModelLoader(config_path=str(self.config.cfg_path))
            self.embeddings = None

            log.info(
                "IndexBuilder initialized",
                data_pdfs=str(self.data_pdfs),
                index_dir=str(self.index_dir),
            )
        except Exception as e:
            raise DocumentPortalException("IndexBuilder initialization error", e)

    def build_and_save(self, force_rebuild: bool = False) -> str:
        try:
            faiss_path = self.index_dir / "index.faiss"

            if faiss_path.exists() and not force_rebuild:
                log.info("FAISS exists; skipping build", path=str(faiss_path))
                return str(self.index_dir)

            if self.embeddings is None:
                log.info("Loading embeddings for index build")
                self.embeddings = self.model_loader.load_embeddings()
                log.info("Embeddings ready")

            pdf_files = sorted(self.data_pdfs.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDFs found in: {self.data_pdfs}")

            docs = []
            bad_pdfs: List[str] = []

            # Tune worker count:
            # - Start with 8
            # - If your disk/antivirus becomes the bottleneck, reduce to 4
            # - If you have a fast NVMe and many CPU cores, you can try 12–16
            max_workers = int(self.config.get_models_param("pdf_load_workers", 8))

            log.info(
                "FAISS build starting",
                pdf_dir=str(self.data_pdfs),
                pdf_count=len(pdf_files),
                max_workers=max_workers,
            )

            # Parallel PDF loading
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_load_one_pdf, pdf): pdf for pdf in pdf_files}

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading PDFs"):
                    loaded_docs, err = fut.result()
                    if loaded_docs:
                        docs.extend(loaded_docs)
                    if err:
                        pdf, e1, e2 = err
                        bad_pdfs.append(pdf)
                        log.warning(
                            "Skipping PDF due to parse error",
                            pdf=pdf,
                            pymupdf_error=e1,
                            pypdf_error=e2,
                        )

            if not docs:
                raise RuntimeError("No documents could be loaded from PDFs.")

            if bad_pdfs:
                log.warning("Some PDFs were skipped", skipped=len(bad_pdfs), first=bad_pdfs[0])

            chunk_size = int(self.config.get_rag_param("chunk_size") or 900)
            chunk_overlap = int(self.config.get_rag_param("chunk_overlap") or 150)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            chunks = splitter.split_documents(docs)

            log.info(
                "Building FAISS index",
                docs=len(docs),
                chunks=len(chunks),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(self.index_dir))

            log.info("FAISS index built and saved", index_dir=str(self.index_dir))
            return str(self.index_dir)

        except Exception as e:
            raise DocumentPortalException("FAISS build error", e)


def main(argv: list[str]) -> int:
    # Usage:
    #   python .\src\build_index.py
    #   python .\src\build_index.py .\config\config.yaml
    #   python .\src\build_index.py .\config\config.yaml --force
    config_path = "config/config.yaml"
    force = False

    for a in argv[1:]:
        if a.strip().lower() == "--force":
            force = True
        else:
            config_path = a

    builder = IndexBuilder(config_path=config_path)
    out = builder.build_and_save(force_rebuild=force)
    print("Index saved to:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
