# src/core_pipeline.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import yaml
from tqdm import tqdm

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

try:
    from langchain_ollama import OllamaLLM as Ollama  # type: ignore
except Exception:
    from langchain_community.llms import Ollama  # type: ignore

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY, PromptType


# ===============================================================
# FUNDER REGISTRY (NIH default)
# ===============================================================
@dataclass(frozen=True)
class FunderSpec:
    key: str
    template_md: Path
    docx_template: Path
    dmptool_template_title: str
    prompt_type_value: str
    retrieval_hint: str


FUNDER_SPECS: Dict[str, FunderSpec] = {
    "NIH": FunderSpec(
        key="NIH",
        template_md=Path("data/inputs/dmp-template.md"),
        docx_template=Path("data/inputs/nih-dms-plan-template.docx"),
        dmptool_template_title="NIH Data Management and Sharing Plan",
        prompt_type_value=PromptType.NIH_DMP.value,
        retrieval_hint="NIH Data Management and Sharing Plan (DMSP) guidance",
    ),
}


def get_funder_spec(funding_agency: str) -> FunderSpec:
    key = (funding_agency or "NIH").strip().upper()
    return FUNDER_SPECS.get(key, FUNDER_SPECS["NIH"])


# ===============================================================
# CONFIGURATION MANAGER (root_dir aware)
# ===============================================================
class ConfigManager:
    """
    Reads config/config.yaml and resolves relative paths using:
      1) config.root_dir (if present)
      2) repo_root (parent of /src)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        repo_root = Path(__file__).resolve().parents[1]
        raw = Path(config_path)
        path = raw if raw.is_absolute() else (repo_root / raw).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.repo_root = repo_root
        self.config_path = path

        self.root_dir = self._resolve_root_dir(self.cfg.get("root_dir"))
        self.paths = self.cfg.get("paths", {}) or {}
        self.models = self.cfg.get("models", {}) or {}
        self.rag = self.cfg.get("rag", {}) or {}

        log.info("Config loaded successfully", config=str(path), root_dir=str(self.root_dir))

    def _resolve_root_dir(self, root_dir_value: Optional[str]) -> Path:
        if not root_dir_value:
            return self.repo_root

        p = Path(str(root_dir_value)).expanduser()
        if p.is_absolute():
            return p.resolve()

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

    def get_model(self, key: str):
        return self.models.get(key)

    def get_rag_param(self, key: str):
        return self.rag.get(key)


# ===============================================================
# MAIN PIPELINE
# ===============================================================
class DMPPipeline:
    def __init__(self, config_path: str = "config/config.yaml", force_rebuild_index: bool = False):
        try:
            self.config = ConfigManager(config_path)
            self.force_rebuild_index = force_rebuild_index

            # Paths from YAML (root_dir aware)
            self.data_pdfs = self.config.get_path("data_pdfs")
            self.index_dir = self.config.get_path("index_dir")

            # Debug/aux output folders (core writes ONLY debug retrieval context)
            self.output_debug = self.config.resolve_path("data/outputs/debug")

            for p in [self.index_dir, self.output_debug]:
                p.mkdir(parents=True, exist_ok=True)

            # Models
            self.model_loader = ModelLoader(config_path=str(self.config.config_path))
            self.embeddings = None

            self.llm_name = self.model_loader.llm_name
            self.llm = Ollama(model=self.llm_name)

            enabled_val = self.config.get_rag_param("enabled")
            self.use_rag_default = True if enabled_val is None else bool(enabled_val)

            self._no_rag_chain_cache: Dict[str, object] = {}
            self.vectorstore = None
            self.retriever = None

            self.last_run_stem: Optional[str] = None

            log.info(
                "DMPPipeline initialized",
                llm=self.llm_name,
                rag_default=self.use_rag_default,
                data_pdfs=str(self.data_pdfs),
                index_dir=str(self.index_dir),
                debug_dir=str(self.output_debug),
                force_rebuild_index=self.force_rebuild_index,
            )

        except Exception as e:
            log.error("Failed to initialize DMPPipeline", error=str(e))
            raise DocumentPortalException("Pipeline initialization error", e)

    # -------------------------
    # Helpers
    # -------------------------
    def _to_bool(self, v, default: Optional[bool] = None) -> Optional[bool]:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            if s in {"false", "0", "no", "n", "off"}:
                return False
        return default

    def _safe_stem(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r'[\\/*?:"<>|]', "_", s)
        s = re.sub(r"\s+", "_", s).strip("_")
        return s or "request"

    def _llm_tag(self) -> str:
        return re.sub(r'[\\/*?:"<>|\s]', "_", (self.llm_name or "").strip()).strip() or "llm"

    def _run_suffix(self, use_rag_final: bool, top_k: int) -> str:
        mode_tag = "rag" if use_rag_final else "norag"
        llm_tag = self._llm_tag()
        if use_rag_final:
            return f"__{mode_tag}__k{top_k}__{llm_tag}"
        return f"__{mode_tag}__{llm_tag}"

    def _format_docs(self, docs) -> str:
        if not docs:
            return ""
        formatted: List[str] = []
        for i, d in enumerate(docs, start=1):
            page = d.metadata.get("page", "")
            src = d.metadata.get("source", "")
            content = (d.page_content or "").strip()
            if not content:
                continue
            formatted.append(f"[CTX-{i}] Page {page} | {src}\n{content}")
        return "\n\n".join(formatted)

    def _trim_context(self, context_text: str, max_chars: int) -> str:
        context_text = (context_text or "").strip()
        if not context_text:
            return ""
        if len(context_text) <= max_chars:
            return context_text
        return context_text[:max_chars].rstrip() + "\n\n[CTX-TRUNCATED]"

    def _select_top_chunks_simple(self, docs, max_chars: int) -> Tuple[List, int]:
        selected = []
        total = 0
        for d in (docs or []):
            txt = (d.page_content or "").strip()
            if not txt:
                continue
            n = len(txt)
            if total + n > max_chars and selected:
                break
            selected.append(d)
            total += n
            if total >= max_chars:
                break
        return selected, total

    def _get_prompt_template(self, spec: FunderSpec):
        try:
            return PROMPT_REGISTRY[spec.prompt_type_value]
        except Exception:
            return PROMPT_REGISTRY[PromptType.NIH_DMP.value]

    def _load_template_text(self, spec: FunderSpec) -> str:
        # resolve template relative to root_dir for consistency
        path = self.config.resolve_path(spec.template_md)
        if path.exists():
            return path.read_text(encoding="utf-8")

        nih_path = self.config.resolve_path(FUNDER_SPECS["NIH"].template_md)
        if nih_path.exists():
            log.warning("Funder template missing; falling back to NIH template", missing=str(path))
            return nih_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Template not found for funder: {spec.key} (missing {path})")

    def _maybe_override_llm(self, llm_model_name: Optional[str]) -> None:
        name = (llm_model_name or "").strip()
        if not name:
            return
        if (self.llm_name or "").strip() == name:
            return
        self.llm_name = name
        self.llm = Ollama(model=self.llm_name)
        log.info("LLM overridden from inputs", llm=self.llm_name)

    def _build_no_rag_chain(self, prompt_template):
        try:
            chain = (
                RunnableMap({"context": lambda x: "", "question": lambda x: x["input"]})
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            log.info("No-RAG chain built successfully", llm=self.llm_name)
            return chain
        except Exception as e:
            raise DocumentPortalException("No-RAG chain build error", e)

    def _get_no_rag_chain_for_funder(self, spec: FunderSpec):
        if spec.key in self._no_rag_chain_cache:
            return self._no_rag_chain_cache[spec.key]
        prompt_template = self._get_prompt_template(spec)
        chain = self._build_no_rag_chain(prompt_template)
        self._no_rag_chain_cache[spec.key] = chain
        return chain

    def _retrieve(self, query: str):
        if self.retriever is None:
            return []
        try:
            return self.retriever.invoke(query) or []
        except Exception:
            try:
                return self.retriever.get_relevant_documents(query) or []
            except Exception:
                fn = getattr(self.retriever, "_get_relevant_documents", None)
                return fn(query) if callable(fn) else []

    # -------------------------
    # Index build/load (FAISS) (kept, still available)
    # -------------------------
    def _load_or_build_index(self, force_rebuild: bool = False):
        try:
            if self.embeddings is None:
                raise RuntimeError("Embeddings are not loaded. Call _ensure_rag_ready() first.")

            faiss_path = self.index_dir / "index.faiss"

            if faiss_path.exists() and not force_rebuild:
                try:
                    log.info("Loading existing FAISS index", path=str(faiss_path))
                    return FAISS.load_local(
                        str(self.index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except Exception as e:
                    log.warning("Failed to load FAISS index. Rebuilding.", reason=str(e))

            pdf_files = sorted(self.data_pdfs.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDFs found in: {self.data_pdfs}")

            docs = []
            bad_pdfs = []

            log.info("FAISS build starting", pdf_dir=str(self.data_pdfs), pdf_count=len(pdf_files))

            for pdf in tqdm(pdf_files, desc="Loading PDFs"):
                try:
                    docs.extend(PyMuPDFLoader(str(pdf)).load())
                except Exception as e1:
                    try:
                        docs.extend(PyPDFLoader(str(pdf)).load())
                    except Exception as e2:
                        bad_pdfs.append(str(pdf))
                        log.warning(
                            "Skipping PDF due to parse error",
                            pdf=str(pdf),
                            pymupdf_error=str(e1),
                            pypdf_error=str(e2),
                        )

            if not docs:
                raise RuntimeError("No documents could be loaded from PDFs.")

            if bad_pdfs:
                log.warning("Some PDFs were skipped", skipped=len(bad_pdfs), first=bad_pdfs[0])

            chunk_size = self.config.get_rag_param("chunk_size") or 900
            chunk_overlap = self.config.get_rag_param("chunk_overlap") or 150

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )
            chunks = splitter.split_documents(docs)

            log.info(
                "Building new FAISS index",
                docs=len(docs),
                chunks=len(chunks),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(self.index_dir))
            log.info("FAISS index built and saved", path=str(self.index_dir))
            return vectorstore

        except Exception as e:
            raise DocumentPortalException("FAISS index error", e)

    # -------------------------
    # NEW: Load-only FAISS (no PDFs)
    # -------------------------
    def _load_index_only(self):
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not loaded. Call _ensure_rag_ready() first.")

        faiss_path = self.index_dir / "index.faiss"
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {faiss_path}. "
                "Build it first using: python .\\src\\build_index.py --force"
            )

        log.info("Loading existing FAISS index (load-only)", path=str(faiss_path))
        return FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    # -------------------------
    # Lazy init RAG components
    # -------------------------
    def _ensure_rag_ready(self):
        if self.retriever is not None and self.vectorstore is not None:
            log.info("RAG components already ready", index_dir=str(self.index_dir))
            return

        if self.embeddings is None:
            log.info("Loading embeddings for RAG", llm=self.llm_name)
            self.embeddings = self.model_loader.load_embeddings()
            log.info("Embeddings ready")

        # NEW: index_mode switch (default keeps original behavior)
        # - "load" => ONLY load index (no PDFs)
        # - "build_or_load" => original: load if exists else build from PDFs
        index_mode = (self.config.get_rag_param("index_mode") or "build_or_load").strip().lower()

        log.info(
            "Preparing FAISS vectorstore",
            index_dir=str(self.index_dir),
            index_mode=index_mode,
            force_rebuild=self.force_rebuild_index,
        )

        if index_mode == "load":
            self.vectorstore = self._load_index_only()
        else:
            self.vectorstore = self._load_or_build_index(force_rebuild=self.force_rebuild_index)

        top_k = self.config.get_rag_param("retriever_top_k") or 6
        use_mmr = self._to_bool(self.config.get_rag_param("use_mmr"), default=True)

        if use_mmr:
            fetch_k = max(20, int(top_k) * 4)
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": int(top_k), "fetch_k": int(fetch_k), "lambda_mult": 0.5},
            )
            log.info("RAG retriever initialized", search_type="mmr", top_k=top_k, fetch_k=fetch_k, lambda_mult=0.5)
        else:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": int(top_k)})
            log.info("RAG retriever initialized", search_type="similarity", top_k=top_k)

    # -------------------------
    # MAIN: Generate DMP (Markdown only)
    # -------------------------
    def generate_dmp(
        self,
        title: Optional[str],
        form_inputs: dict,
        use_rag: Optional[bool] = None,
        funding_agency: str = "NIH",
        llm_model_name: Optional[str] = None,
    ) -> str:
        try:
            spec = get_funder_spec(funding_agency)
            funding_agency = spec.key
            log.info(
                "Funding agency selected",
                funding_agency=funding_agency,
                prompt_type=spec.prompt_type_value,
                template_md=str(spec.template_md),
                retrieval_hint=spec.retrieval_hint,
            )

            self._maybe_override_llm(llm_model_name)

            prompt_template = self._get_prompt_template(spec)
            template_text = self._load_template_text(spec)

            use_rag_final = self._to_bool(use_rag, default=self.use_rag_default)

            log.info(
                "Pipeline mode selected",
                mode=("rag" if use_rag_final else "norag"),
                use_rag_input=use_rag,
                rag_default=self.use_rag_default,
                use_rag_final=use_rag_final,
            )

            title_clean = (title or "").strip()

            user_elements: List[str] = []
            for key, val in (form_inputs or {}).items():
                if val is None:
                    continue

                if isinstance(val, str):
                    v = val.strip()
                    if not v:
                        continue
                    user_elements.append(f"{key.replace('_', ' ').title()}: {v}")
                else:
                    try:
                        txt = json.dumps(val, ensure_ascii=False)
                    except Exception:
                        txt = str(val)
                    txt = (txt or "").strip()
                    if txt:
                        user_elements.append(f"{key.replace('_', ' ').title()}: {txt}")

            log.info(
                "Inputs parsed",
                title_present=bool(title_clean),
                title=(title_clean if title_clean else None),
                input_fields=len(user_elements),
            )

            top_k = int(self.config.get_rag_param("retriever_top_k") or 6)
            base_stem = self._safe_stem(title_clean)
            safe_stem = f"{base_stem}{self._run_suffix(use_rag_final, top_k=top_k)}"
            self.last_run_stem = safe_stem

            log.info(
                "Run naming",
                run_stem=safe_stem,
                base_stem=base_stem,
                mode=("rag" if use_rag_final else "norag"),
                top_k=top_k,
                llm=self.llm_name,
            )

            rag_usage_rules = (
                f"IMPORTANT (RAG MODE): You MUST use the provided context as authoritative {funding_agency} guidance. "
                "Incorporate specific details from it, and prefer it over generic wording.\n\n"
            )

            header_line = (
                f"Create a complete {funding_agency} Data Management plan.\n\n"
                if not title_clean
                else f"Create a complete {funding_agency} Data Management plan for the project '{title_clean}'.\n\n"
            )

            question_text = (
                header_line
                + (rag_usage_rules if use_rag_final else "")
                + "User Inputs:\n"
                + ("\n".join(user_elements) if user_elements else "(none provided)")
                + "\n\n"
                + f"Use the following {funding_agency} Markdown template. Do not alter section titles:\n"
                + template_text
            )

            log.info(
                "Prompt scaffold ready",
                question_chars=len(question_text),
                template_chars=len(template_text),
                user_elements_lines=len(user_elements),
            )

            if use_rag_final:
                log.info("RAG phase start", step="ensure_rag_ready")
                self._ensure_rag_ready()
                log.info("RAG phase ready", step="ensure_rag_ready")

                retrieval_query = (
                    f"{spec.retrieval_hint} relevant to this project.\n"
                    + (f"Project title: {title_clean}\n" if title_clean else "")
                    + "User Inputs:\n"
                    + ("\n".join(user_elements) if user_elements else "(none provided)")
                    + "\n"
                )

                log.info("Retrieval start", query_chars=len(retrieval_query), top_k=top_k)
                docs = self._retrieve(retrieval_query) or []
                log.info("Retrieval raw done", retrieved_docs=len(docs))

                max_ctx_chars = int(self.config.get_rag_param("max_context_chars") or 12000)
                select_first = self._to_bool(self.config.get_rag_param("select_top_chunks"), default=True)

                if select_first:
                    docs, selected_chars = self._select_top_chunks_simple(docs, max_chars=max_ctx_chars)
                    log.info(
                        "Chunk selection",
                        strategy="first_fit",
                        selected_docs=len(docs),
                        approx_selected_chars=selected_chars,
                        max_ctx_chars=max_ctx_chars,
                    )

                docs = docs[:top_k]

                context_text = self._format_docs(docs)
                context_text = self._trim_context(context_text, max_chars=max_ctx_chars)

                debug_path = self.output_debug / f"{safe_stem}.retrieved_context.txt"
                debug_path.write_text(context_text, encoding="utf-8")

                log.info(
                    "Retrieval complete",
                    retriever_top_k=top_k,
                    retrieved_docs=len(docs),
                    retrieved_chars=len(context_text),
                    max_ctx_chars=max_ctx_chars,
                    debug_context_file=str(debug_path),
                )

                full_prompt = prompt_template.format(context=context_text, question=question_text)

                log.info("LLM start", mode="rag", prompt_chars=len(full_prompt))
                result = self.llm.invoke(full_prompt)
                log.info("LLM done", mode="rag")

            else:
                full_prompt = prompt_template.format(context="", question=question_text)
                log.info("LLM start", mode="norag", prompt_chars=len(full_prompt))
                result = self.llm.invoke(full_prompt)
                log.info("LLM done", mode="norag")

            if not isinstance(result, str):
                result = str(result)

            log.info("DMP generation finished", mode=("rag" if use_rag_final else "norag"), output_chars=len(result))
            return result

        except Exception as e:
            raise DocumentPortalException("DMP generation error", e)
