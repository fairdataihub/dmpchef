# ===============================================================
# prompt_library.py — NIH DMP Generation Prompts (Aligned with Notebook)
# ===============================================================

from langchain_core.prompts import PromptTemplate
from enum import Enum


class PromptType(Enum):
    NIH_DMP = "nih_dmp"


PROMPT_REGISTRY = {
    PromptType.NIH_DMP.value: PromptTemplate(
        template="""You are an expert biomedical data steward and grant writer.
Create a high-quality NIH Data Management and Sharing Plan (DMSP)
based on the retrieved NIH context and the user's query.

----
Context from NIH Repository:
{context}

----
Question:
{question}

Use the context above and follow the NIH template structure. 
Write fluently, cohesively, and in professional Markdown format. 
Ensure each section matches NIH Data Management Plan expectations 
and maintains the same structure as provided in the DMP Markdown template.
""",
        input_variables=["context", "question"],
    )
}
