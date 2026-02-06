
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

from utils.dmptool_json import build_dmptool_json
from utils.nih_docx_writer import build_nih_docx_from_template


# ===============================================================
# FUNDER REGISTRY (Funder-ready; NIH is default)
# ===============================================================
@dataclass(frozen=True)
class FunderSpec:
    key: str
    template_md: Path
    docx_template: Path
    dmptool_template_title: str
    prompt_type_value: str
    role_label: str
    retrieval_hint: str


FUNDER_SPECS: Dict[str, FunderSpec] = {
    "NIH": FunderSpec(
        key="NIH",
        template_md=Path("data/inputs/dmp-template.md"),
        docx_template=Path("data/inputs/nih-dms-plan-template.docx"),
        dmptool_template_title="NIH Data Management and Sharing Plan",
        prompt_type_value=PromptType.NIH_DMP.value,
        role_label="NIH data steward and grant writer",
        retrieval_hint="NIH Data Management and Sharing Plan (DMSP) guidance",
    ),
}


def get_funder_spec(funding_agency: str) -> FunderSpec:
    key = (funding_agency or "NIH").strip().upper()
    return FUNDER_SPECS.get(key, FUNDER_SPECS["NIH"])


# ===============================================================
# CONFIGURATION MANAGER
# ===============================================================
class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.paths = self.cfg.get("paths", {}) or {}
        self.models = self.cfg.get("models", {}) or {}
        self.rag = self.cfg.get("rag", {}) or {}

        log.info("✅ Config loaded successfully")

    def get_path(self, key: str) -> Path:
        val = self.paths.get(key)
        if not val:
            raise KeyError(f"❌ Missing config.paths.{key} in YAML")
        return Path(val)

    def get_model(self, key: str):
        return self.models.get(key)

    def get_rag_param(self, key: str):
        return self.rag.get(key)


# ===============================================================
# MAIN PIPELINE CLASS
# ===============================================================
class DMPPipeline:
    def __init__(self, config_path: str = "config/config.yaml", force_rebuild_index: bool = False):
        try:
            self.config = ConfigManager(config_path)
            self.force_rebuild_index = force_rebuild_index

            self.data_pdfs = self.config.get_path("data_pdfs")
            self.index_dir = self.config.get_path("index_dir")
            self.output_md = self.config.get_path("output_md")
            self.output_docx = self.config.get_path("output_docx")

            self.output_json = Path("data/outputs/json")
            self.output_debug = Path("data/outputs/debug")

            for p in [self.output_md, self.output_docx, self.output_json, self.output_debug, self.index_dir]:
                p.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()
            self.embeddings = None

            self.llm_name = self.model_loader.llm_name
            self.llm = Ollama(model=self.llm_name)

            enabled_val = self.config.get_rag_param("enabled")
            self.use_rag_default = True if enabled_val is None else bool(enabled_val)

            self._no_rag_chain_cache: Dict[str, object] = {}

            self.vectorstore = None
            self.retriever = None

            # ✅ NEW: main.py can read this to name outputs with __rag__/__norag__
            self.last_run_stem: Optional[str] = None

            log.info("✅ DMPPipeline initialized", llm=self.llm_name, rag_default=self.use_rag_default)

        except Exception as e:
            log.error("❌ Failed to initialize DMPPipeline", error=str(e))
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

    def _safe_title(self, title: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "_", (title or "").strip()).strip()

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
        path = spec.template_md
        if path.exists():
            return path.read_text(encoding="utf-8")

        nih_path = FUNDER_SPECS["NIH"].template_md
        if nih_path.exists():
            log.warning("⚠️ Funder template missing; falling back to NIH template", missing=str(path))
            return nih_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"❌ Template not found for funder: {spec.key} (missing {path})")

    def _build_no_rag_chain(self, prompt_template):
        try:
            chain = (
                RunnableMap({"context": lambda x: "", "question": lambda x: x["input"]})
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            log.info("🔗 No-RAG chain built successfully", llm=self.llm_name)
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

    # ✅ NEW: LangChain-version-safe retrieval
    def _retrieve(self, query: str):
        """
        New LangChain retrievers: retriever.invoke(query)
        Older LangChain: retriever.get_relevant_documents(query)
        """
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
    # Index build/load (FAISS)
    # -------------------------
    def _load_or_build_index(self, force_rebuild: bool = False):
        try:
            if self.embeddings is None:
                raise RuntimeError("Embeddings are not loaded. Call _ensure_rag_ready() first.")

            faiss_path = self.index_dir / "index.faiss"

            if faiss_path.exists() and not force_rebuild:
                try:
                    log.info("📦 Loading existing FAISS index", path=str(faiss_path))
                    return FAISS.load_local(
                        str(self.index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except Exception as e:
                    log.warning(f"⚠️ Failed to load FAISS index. Rebuilding... Reason: {e}")

            pdf_files = sorted(self.data_pdfs.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"❌ No PDFs found in: {self.data_pdfs}")

            docs = []
            bad_pdfs = []

            for pdf in tqdm(pdf_files, desc="📥 Loading PDFs"):
                try:
                    docs.extend(PyMuPDFLoader(str(pdf)).load())
                except Exception as e1:
                    try:
                        docs.extend(PyPDFLoader(str(pdf)).load())
                    except Exception as e2:
                        bad_pdfs.append(str(pdf))
                        log.warning(
                            "⚠️ Skipping PDF due to parse error",
                            pdf=str(pdf),
                            pymupdf_error=str(e1),
                            pypdf_error=str(e2),
                        )

            if not docs:
                raise RuntimeError("❌ No documents could be loaded from PDFs.")

            if bad_pdfs:
                log.warning("⚠️ Some PDFs were skipped", skipped=len(bad_pdfs))
                log.warning("First skipped PDF", first=bad_pdfs[0])

            chunk_size = self.config.get_rag_param("chunk_size") or 800
            chunk_overlap = self.config.get_rag_param("chunk_overlap") or 120

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(docs)

            log.info("🧱 Building new FAISS index ...", chunks=len(chunks))
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(self.index_dir))
            log.info("✅ FAISS index built and saved", path=str(self.index_dir))
            return vectorstore

        except Exception as e:
            raise DocumentPortalException("FAISS index error", e)

    # -------------------------
    # Lazy init RAG components
    # -------------------------
    def _ensure_rag_ready(self):
        if self.retriever is not None and self.vectorstore is not None:
            return

        if self.embeddings is None:
            self.embeddings = self.model_loader.load_embeddings()

        self.vectorstore = self._load_or_build_index(force_rebuild=self.force_rebuild_index)

        top_k = self.config.get_rag_param("retriever_top_k") or 6
        use_mmr = self._to_bool(self.config.get_rag_param("use_mmr"), default=True)

        if use_mmr:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": max(20, top_k * 4), "lambda_mult": 0.5},
            )
            log.info("✅ RAG retriever initialized", type="mmr", top_k=top_k)
        else:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            log.info("✅ RAG retriever initialized", type="similarity", top_k=top_k)

    def _cleanup_title_json(self, safe_title: str):
        keep = f"{safe_title}.dmptool.json"
        for p in self.output_json.glob(f"{safe_title}*.json"):
            if p.name != keep:
                try:
                    p.unlink()
                except Exception:
                    pass

    # -------------------------
    # MAIN: Generate DMP
    # -------------------------
    def generate_dmp(
        self,
        title: str,
        form_inputs: dict,
        use_rag: Optional[bool] = None,
        funding_agency: str = "NIH",
    ):
        try:
            title = (title or "").strip()
            if not title:
                raise ValueError("❌ Title is required.")

            spec = get_funder_spec(funding_agency)
            funding_agency = spec.key
            log.info("🏷️ Funding agency selected", funding_agency=funding_agency)

            prompt_template = self._get_prompt_template(spec)
            template_text = self._load_template_text(spec)

            use_rag_final = self._to_bool(use_rag, default=self.use_rag_default)
            log.info("🧾 RAG decision", use_rag_input=use_rag, rag_default=self.use_rag_default, use_rag_final=use_rag_final)

            user_elements = [
                f"{key.replace('_', ' ').title()}: {val}".strip()
                for key, val in (form_inputs or {}).items()
                if isinstance(val, str) and val.strip()
            ]

            top_k = int(self.config.get_rag_param("retriever_top_k") or 6)

            base_title = self._safe_title(title)
            safe_title = f"{base_title}{self._run_suffix(use_rag_final, top_k=top_k)}"

            # ✅ NEW: expose this to main.py so it saves files with suffix
            self.last_run_stem = safe_title

            rag_usage_rules = (
                f"IMPORTANT (RAG MODE): You MUST use the provided context as authoritative {funding_agency} guidance. "
                "Incorporate specific details from it, and prefer it over generic wording.\n\n"
            )

            question_text = (
                f"You are an expert {spec.role_label}. "
                f"Create a complete {funding_agency} Data Management plan "
                f"for the project titled '{title}'.\n\n"
                f"{rag_usage_rules if use_rag_final else ''}"
                f"User Inputs:\n{chr(10).join(user_elements)}\n\n"
                f"Use the following {funding_agency} Markdown template. Do not alter section titles:\n"
                f"{template_text}"
            )

            if use_rag_final:
                self._ensure_rag_ready()

                retrieval_query = (
                    f"{spec.retrieval_hint} relevant to this project.\n"
                    f"Project title: {title}\n"
                    f"User Inputs:\n{chr(10).join(user_elements)}\n"
                )

                # ✅ FIX: version-safe retrieval
                docs = self._retrieve(retrieval_query) or []

                max_ctx_chars = int(self.config.get_rag_param("max_context_chars") or 12000)
                select_first = self._to_bool(self.config.get_rag_param("select_top_chunks"), default=True)

                if select_first:
                    docs, _ = self._select_top_chunks_simple(docs, max_chars=max_ctx_chars)

                docs = docs[:top_k]

                context_text = self._format_docs(docs)
                context_text = self._trim_context(context_text, max_chars=max_ctx_chars)

                debug_path = self.output_debug / f"{safe_title}.retrieved_context.txt"
                debug_path.write_text(context_text, encoding="utf-8")

                log.info(
                    "🔎 Retrieval complete",
                    retriever_top_k=top_k,
                    retrieved_docs=len(docs),
                    retrieved_chars=len(context_text),
                    debug_context_file=str(debug_path),
                )

                full_prompt = prompt_template.format(context=context_text, question=question_text)

                result = self.llm.invoke(full_prompt)
                if not isinstance(result, str):
                    result = str(result)

            else:
                no_rag_chain = self._get_no_rag_chain_for_funder(spec)
                result = no_rag_chain.invoke({"input": question_text})
                if not isinstance(result, str):
                    result = str(result)

            md_path = self.output_md / f"{safe_title}.md"
            docx_path = self.output_docx / f"{safe_title}.docx"
            dmptool_json_path = self.output_json / f"{safe_title}.dmptool.json"

            md_path.write_text(result, encoding="utf-8")

            docx_template_path = spec.docx_template if spec.docx_template.exists() else FUNDER_SPECS["NIH"].docx_template
            if not spec.docx_template.exists():
                log.warning("⚠️ Funder DOCX template missing; falling back to NIH template", missing=str(spec.docx_template))

            build_nih_docx_from_template(
                template_docx_path=str(docx_template_path),
                output_docx_path=str(docx_path),
                project_title=title,
                generated_markdown=result,
            )

            dmptool_obj = build_dmptool_json(
                template_title=spec.dmptool_template_title,
                project_title=title,
                form_inputs=form_inputs,
                generated_markdown=result,
                provenance="dmpchef",
            )

            self._cleanup_title_json(safe_title)

            with open(dmptool_json_path, "w", encoding="utf-8") as f:
                json.dump(dmptool_obj, f, indent=2, ensure_ascii=False)

            log.info(
                "✅ DMP generated successfully",
                title=title,
                funding_agency=funding_agency,
                use_rag=use_rag_final,
                md=str(md_path),
                docx=str(docx_path),
                dmptool_json=str(dmptool_json_path),
            )
            return result

        except Exception as e:
            raise DocumentPortalException("DMP generation error", e)
