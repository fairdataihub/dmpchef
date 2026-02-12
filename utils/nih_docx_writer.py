# utils/nih_docx_writer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


PROMPT_ANCHORS: List[Tuple[str, List[str]]] = [
    ("e1_q1", ["Summarize the types and estimated amount of scientific data expected to be generated in the project"]),
    ("e1_q2", ["Describe which scientific data from the project will be preserved and shared and provide the rationale for this decision"]),
    ("e1_q3", ["Briefly list the metadata, other relevant data, and any associated documentation"]),
    ("e2_q1", ["State whether specialized tools, software, and/or code are needed to access or manipulate shared scientific data"]),
    ("e3_q1", ["State what common data standards will be applied to the scientific data and associated metadata to enable interoperability of datasets and resources"]),
    ("e4_q1", ["Provide the name of the repository(ies) where scientific data and metadata arising from the project will be archived"]),
    ("e4_q2", ["Describe how the scientific data will be findable and identifiable"]),
    ("e4_q3", ["Describe when the scientific data will be made available to other users"]),
    ("e5_q1", ["NIH expects that in drafting Plans, researchers maximize the appropriate sharing of scientific data."]),
    ("e5_q2", ["State whether access to the scientific data will be controlled"]),
    ("e5_q3", ["If generating scientific data derived from humans, describe how the privacy, rights, and confidentiality of human research participants will be protected"]),
    ("e6_q1", ["Describe how compliance with this Plan will be monitored and managed, frequency of oversight, and by whom at your institution"]),
]


# ============================================================
# Text helpers
# ============================================================
def _clean(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _clean_answer(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    # trim common footers
    text = re.split(r"(?mi)^\s*---\s*$", text, maxsplit=1)[0]
    text = re.split(r"(?mi)^\s*please note\b.*$", text, maxsplit=1)[0]
    text = re.split(r"(?mi)^\s*i hope\b.*$", text, maxsplit=1)[0]
    return text.strip()


# ============================================================
# DOCX helpers
# ============================================================
def _insert_paragraph_after(paragraph: Paragraph, text: str = "", style_name: Optional[str] = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)

    if style_name:
        try:
            new_para.style = style_name
        except Exception:
            pass

    if text:
        new_para.add_run(text)

    return new_para


def _delete_paragraph(paragraph: Paragraph) -> None:
    p = paragraph._element
    p.getparent().remove(p)
    paragraph._p = paragraph._element = None  # type: ignore


def _find_anchor_paragraph_index(doc: Document, anchor_aliases: List[str]) -> Optional[int]:
    aliases_norm = [_norm_ws(a) for a in (anchor_aliases or []) if (a or "").strip()]
    if not aliases_norm:
        return None

    for i, p in enumerate(doc.paragraphs):
        pt = _norm_ws(p.text or "")
        if not pt:
            continue
        for a in aliases_norm:
            if a and a in pt:
                return i
    return None


def _remove_placeholder_only_paragraphs(doc: Document) -> None:
    placeholders = {k.lower() for k, _ in PROMPT_ANCHORS}
    for p in doc.paragraphs:
        if (p.text or "").strip().lower() in placeholders:
            p.text = ""


def _copy_paragraph_format(dst: Paragraph, src: Paragraph) -> None:
    """
    SAFE: src may be detached (deleted). If so, skip.
    """
    try:
        if getattr(src, "_element", None) is None:
            return
        dpf = dst.paragraph_format
        spf = src.paragraph_format

        dpf.left_indent = spf.left_indent
        dpf.right_indent = spf.right_indent
        dpf.first_line_indent = spf.first_line_indent

        dpf.space_before = spf.space_before
        dpf.space_after = spf.space_after
        dpf.line_spacing = spf.line_spacing

        dpf.alignment = spf.alignment
        dpf.keep_together = spf.keep_together
        dpf.keep_with_next = spf.keep_with_next
        dpf.page_break_before = spf.page_break_before
        dpf.widow_control = spf.widow_control
    except Exception:
        return


_INSTRUCTION_START_RE = re.compile(r"(?i)^\s*(summarize|describe|provide|state|indicate|explain)\b")


def _is_instruction_paragraph(text: str) -> bool:
    t = _clean(text)
    if not t:
        return False
    return bool(_INSTRUCTION_START_RE.match(t))


def _remove_instruction_paragraphs_after(doc: Document, anchor_idx: int) -> None:
    """
    Delete consecutive instruction paragraphs immediately after the prompt.
    Stop on blank OR next Element heading OR next label-ish line.
    """
    j = anchor_idx + 1
    while j < len(doc.paragraphs):
        p = doc.paragraphs[j]
        txt = _clean(p.text or "")

        if txt == "":
            break
        if re.match(r"(?i)^\s*element\s+\d+\b", txt):
            break

        if _is_instruction_paragraph(txt) or ("see selecting a data repository" in _norm_ws(txt)):
            _delete_paragraph(p)
            continue

        break


def _add_answer_paragraph_after(
    anchor: Paragraph,
    text: str,
    style_name: Optional[str],
    format_source: Paragraph,
) -> Paragraph:
    p = _insert_paragraph_after(anchor, "", style_name=style_name)
    _copy_paragraph_format(p, format_source)
    if text:
        p.add_run(text)
    return p


def _write_block_after_anchor(doc: Document, anchor_aliases: List[str], answer: str) -> None:
    idx = _find_anchor_paragraph_index(doc, anchor_aliases)
    if idx is None:
        return

    anchor_p = doc.paragraphs[idx]
    _remove_instruction_paragraphs_after(doc, idx)

    style_name = anchor_p.style.name if anchor_p.style else None
    format_source = anchor_p

    # If template has a blank answer line right after the prompt, reuse it
    first_answer_para: Optional[Paragraph] = None
    if idx + 1 < len(doc.paragraphs):
        nxt = doc.paragraphs[idx + 1]
        if (nxt.text or "").strip() == "":
            first_answer_para = nxt
            format_source = nxt
            if nxt.style:
                style_name = nxt.style.name

    answer = _clean_answer(answer)
    if not answer:
        if first_answer_para is not None:
            first_answer_para.text = ""
        else:
            _add_answer_paragraph_after(anchor_p, "", style_name, format_source)
        return

    # Split into paragraphs; normalize list markers into plain paragraphs
    raw_lines = [ln.rstrip() for ln in answer.split("\n")]
    paragraphs: List[str] = []
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf
        joined = " ".join([x.strip() for x in buf if x.strip()]).strip()
        if joined:
            paragraphs.append(joined)
        buf = []

    for ln in raw_lines:
        s = ln.strip()
        if s == "":
            flush_buf()
            continue

        if s.startswith(("- ", "• ", "+ ", "* ")):
            flush_buf()
            paragraphs.append(s[2:].strip())
            continue

        buf.append(s)

    flush_buf()

    if first_answer_para is not None:
        first_answer_para.text = ""
        if paragraphs:
            first_answer_para.add_run(paragraphs[0])
        last_p = first_answer_para
        start_i = 1
    else:
        last_p = anchor_p
        start_i = 0

    for i in range(start_i, len(paragraphs)):
        last_p = _add_answer_paragraph_after(last_p, paragraphs[i], style_name, format_source)
        if i != len(paragraphs) - 1:
            last_p = _add_answer_paragraph_after(last_p, "", style_name, format_source)

    _add_answer_paragraph_after(last_p, "", style_name, format_source)


# ============================================================
# DMPTool JSON -> blocks
# ============================================================
def _extract_blocks_from_dmptool_json(plan_json: Dict[str, Any]) -> Dict[str, str]:
    """
    Expects shape like:
      plan_json["dmptool"]["narrative"]["section"] -> list
      section[i]["order"] -> 1..6
      section[i]["question"] -> list with question[j]["order"] -> 1..3
      question[j]["answer"]["json"]["answer"] -> string
    """
    blocks = {
        "e1_q1": "", "e1_q2": "", "e1_q3": "",
        "e2_q1": "", "e3_q1": "",
        "e4_q1": "", "e4_q2": "", "e4_q3": "",
        "e5_q1": "", "e5_q2": "", "e5_q3": "",
        "e6_q1": "",
    }

    section_list = (
        (plan_json or {})
        .get("dmptool", {})
        .get("narrative", {})
        .get("section", [])
    )
    if not isinstance(section_list, list):
        return blocks

    key_map = {
        (1, 1): "e1_q1", (1, 2): "e1_q2", (1, 3): "e1_q3",
        (2, 1): "e2_q1",
        (3, 1): "e3_q1",
        (4, 1): "e4_q1", (4, 2): "e4_q2", (4, 3): "e4_q3",
        (5, 1): "e5_q1", (5, 2): "e5_q2", (5, 3): "e5_q3",
        (6, 1): "e6_q1",
    }

    for sec in section_list:
        if not isinstance(sec, dict):
            continue
        s_order = sec.get("order")
        q_list = sec.get("question", [])
        if not isinstance(s_order, int) or not isinstance(q_list, list):
            continue

        for q in q_list:
            if not isinstance(q, dict):
                continue
            q_order = q.get("order")
            if not isinstance(q_order, int):
                continue

            ans = (
                (q.get("answer") or {})
                .get("json", {})
                .get("answer", "")
            )
            k = key_map.get((s_order, q_order))
            if k:
                blocks[k] = _clean_answer(ans)

    return blocks


# ============================================================
# Public API
# ============================================================
def build_nih_docx_from_template(
    template_docx_path: str,
    output_docx_path: str,
    project_title: str,  # kept for backward compatibility; not inserted
    generated_markdown: str = "",
    *,
    plan_json: Optional[Dict[str, Any]] = None,
) -> None:
    doc = Document(str(template_docx_path))

    _remove_placeholder_only_paragraphs(doc)

    if plan_json:
        blocks = _extract_blocks_from_dmptool_json(plan_json)
    else:
        # fallback (old behavior): just dump markdown as-is into answers via your markdown parser if needed
        raise ValueError("plan_json is required for DOCX generation in this mode.")

    for key, aliases in PROMPT_ANCHORS:
        answer = (blocks.get(key, "") or "").strip()
        if not answer:
            continue
        _write_block_after_anchor(doc, aliases, answer)

    Path(str(output_docx_path)).parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx_path))
