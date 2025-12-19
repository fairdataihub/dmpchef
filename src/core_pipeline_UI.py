# ===============================================================
# core_pipeline_web.py — Web-Integrated RAG Core (Markdown-Aligned)
# ===============================================================
import re
import json
from pathlib import Path
from tqdm import tqdm
import pypandoc
import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY, PromptType


# ===============================================================
# CONFIGURATION MANAGER
# ===============================================================
class ConfigManager:
    """Loads and provides access to YAML configuration."""

    def __init__(self, config_path="config/config.yaml"):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.paths = self.cfg.get("paths", {})
        self.models = self.cfg.get("models", {})
        self.rag = self.cfg.get("rag", {})

        log.info("✅ Config loaded successfully")

    def get_path(self, key):
        return Path(self.paths.get(key))

    def get_model(self, key):
        return self.models.get(key)

    def get_rag_param(self, key):
        return self.rag.get(key)


# ===============================================================
# MAIN PIPELINE CLASS
# ===============================================================
class DMPPipeline:
    """End-to-end pipeline for NIH DMP generation via web input."""

    def __init__(self, config_path="config/config.yaml"):
        try:
            # --- Load configuration ---
            self.config = ConfigManager(config_path)
            self.data_pdfs = self.config.get_path("data_pdfs")
            self.index_dir = self.config.get_path("index_dir")
            self.template_md = Path("data/inputs/dmp-template.md")
            self.output_md = self.config.get_path("output_md")
            self.output_docx = self.config.get_path("output_docx")
            self.output_json = Path("data/outputs/json")

            for p in [self.output_md, self.output_docx, self.output_json]:
                p.mkdir(parents=True, exist_ok=True)

            # --- Load template file ---
            if not self.template_md.exists():
                raise FileNotFoundError(f"❌ DMP template not found: {self.template_md}")
            self.template_text = self.template_md.read_text(encoding="utf-8")

            # --- Load models ---
            self.model_loader = ModelLoader()
            self.embeddings = self.model_loader.load_embeddings()
            self.llm_name = self.model_loader.llm_name
            self.llm = Ollama(model=self.llm_name)

            # --- Load prompt template from registry ---
            self.prompt_template = PROMPT_REGISTRY[PromptType.NIH_DMP.value]

            log.info("✅ DMPPipeline initialized (Markdown-Aligned Mode)")

        except Exception as e:
            log.error("❌ Failed to initialize DMPPipeline", error=str(e))
            raise DocumentPortalException("Pipeline initialization error", e)

    # ---------------------------------------------------------------
    def _load_or_build_index(self, force_rebuild=False):
        """Load or build FAISS vector index."""
        try:
            faiss_path = self.index_dir / "index.faiss"
            if faiss_path.exists() and not force_rebuild:
                log.info("📦 Loading existing FAISS index", path=str(faiss_path))
                return FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

            pdf_files = sorted(self.data_pdfs.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDFs found in {self.data_pdfs}")

            docs = []
            for pdf in tqdm(pdf_files, desc="📥 Loading PDFs"):
                loader = PyPDFLoader(str(pdf))
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get_rag_param("chunk_size"),
                chunk_overlap=self.config.get_rag_param("chunk_overlap"),
            )
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(self.index_dir))
            log.info("✅ FAISS index built and saved")
            return vectorstore
        except Exception as e:
            raise DocumentPortalException("FAISS index error", e)

    # ---------------------------------------------------------------
    def _build_rag_chain(self, retriever):
        """Build RAG chain using the prompt registry."""
        try:
            rag_chain = (
                RunnableMap({
                    "context": lambda x: retriever.invoke(x["input"]),
                    "question": lambda x: x["input"],
                })
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            log.info("🔗 RAG chain built successfully")
            return rag_chain
        except Exception as e:
            raise DocumentPortalException("RAG chain build error", e)

    # ---------------------------------------------------------------
    def generate_dmp(self, title: str, form_inputs: dict):
        """Generate NIH DMP dynamically from user web form input."""
        try:
            retriever = self._load_or_build_index().as_retriever(
                search_kwargs={"k": self.config.get_rag_param("retriever_top_k")}
            )
            rag_chain = self._build_rag_chain(retriever)

            # --- Combine user-provided form input ---
            user_elements = [
                f"{key.replace('_',' ').title()}: {val}" for key, val in form_inputs.items() if val.strip()
            ]
            query = (
                f"You are an NIH data steward. Create a complete Data Management and Sharing Plan "
                f"for project titled '{title}'. Use the NIH DMP Markdown template below and "
                f"integrate the user's inputs accordingly.\n\n"
                f"User Inputs:\n{chr(10).join(user_elements)}\n\n"
                f"NIH DMP Template:\n{self.template_text}"
            )

            # --- Retrieve context and generate structured text ---
            result = rag_chain.invoke({"input": query})

            # --- Save outputs ---
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title.strip())
            md_path = self.output_md / f"{safe_title}.md"
            docx_path = self.output_docx / f"{safe_title}.docx"
            json_path = self.output_json / f"{safe_title}.json"

            md_path.write_text(result, encoding="utf-8")
            pypandoc.convert_text(result, "docx", format="md", outputfile=str(docx_path))

            json.dump(
                {
                    "title": title,
                    "form_inputs": form_inputs,
                    "template_used": str(self.template_md),
                    "generated_markdown": result,
                },
                open(json_path, "w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )

            log.info("✅ DMP generated successfully (Markdown structure preserved)", title=title)
            return result
        except Exception as e:
            raise DocumentPortalException("DMP generation error", e)
