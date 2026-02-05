# ===============================================================
# prompt_library.py — NIH DMP Generation Prompts
# (works for BOTH RAG and NO-RAG)
# ===============================================================

from enum import Enum
from langchain_core.prompts import PromptTemplate


class PromptType(Enum):
    NIH_DMP = "nih_dmp"


PROMPT_REGISTRY = {
    PromptType.NIH_DMP.value: PromptTemplate(
        template="""You are an expert biomedical data steward and grant writer.

Your task:
Generate a COMPLETE NIH Data Management and Sharing Plan (DMSP) in professional Markdown.

Hard requirements (must follow):
1) You MUST keep the NIH DMSP template section headings exactly as provided in the user's question/template.
2) You MUST include EVERY section heading in the output (no missing sections).
3) Fill every section with content:
   - If the user did not provide enough detail for a section, write a reasonable NIH-appropriate placeholder such as:
     "Not provided by the project at this time" / "To be determined" / "Not applicable".
   - Do NOT leave sections blank.
4) If NIH repository context is provided, use it to improve NIH-aligned wording.
   If the context is empty or unhelpful, rely on the user's inputs and NIH best practices.

----
NIH Repository Context (may be empty in NO-RAG):
{context}

----
User Question / Instructions (includes NIH template + user inputs):
{question}

Now produce the final NIH DMSP in Markdown with all sections present and complete.
""",
        input_variables=["context", "question"],
    )
}
