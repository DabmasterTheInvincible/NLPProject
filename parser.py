#!/usr/bin/env python3
# parser.py  — heading-first, classifier-free sectioner

from __future__ import annotations
import argparse, json, re, sys, os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional PDF parsers
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:
    fitz = None

# ---------------------------- Ontology ---------------------------- #

CANON = [
    "TITLE","ABSTRACT","KEYWORDS","INTRODUCTION","BACKGROUND","RELATED_WORK",
    "METHODS","EXPERIMENTAL","RESULTS","DISCUSSION","CONCLUSION","LIMITATIONS",
    "FUTURE_WORK","ACKNOWLEDGMENTS","REFERENCES","APPENDIX","SUPPLEMENT",
    "FIGURE_CAPTION","TABLE_CAPTION",
]

# Extend this list for your journals/languages
ONTOLOGY: Dict[str, List[str]] = {
    "TITLE": [],
    "ABSTRACT": ["abstract"],
    "KEYWORDS": ["keywords", "index terms"],
    "INTRODUCTION": ["introduction", "overview"],
    "BACKGROUND": ["background"],
    "RELATED_WORK": ["related work", "literature review", "previous work", "state of the art"],
    "METHODS": ["methods", "materials and methods", "methodology", "approach", "experimental setup"],
    "EXPERIMENTAL": ["experimental", "experiments"],
    "RESULTS": ["results", "findings"],
    "DISCUSSION": ["discussion", "analysis and discussion"],
    "CONCLUSION": ["conclusion", "conclusions", "concluding remarks", "summary and conclusion"],
    "LIMITATIONS": ["limitations", "threats to validity"],
    "FUTURE_WORK": ["future work", "outlook", "perspectives", "further research"],
    "ACKNOWLEDGMENTS": ["acknowledgment", "acknowledgments", "acknowledgements", "funding"],
    "REFERENCES": ["references", "bibliography", "works cited"],
    "APPENDIX": ["appendix", "supplementary"],
    "SUPPLEMENT": ["supplement"],
    "FIGURE_CAPTION": ["figure", "fig."],
    "TABLE_CAPTION": ["table"],
}
HEADING_NUMBER_RE = re.compile(r"^(?:[0-9IVX]+(?:\.[0-9]+)*)\s+", re.I)

# ---------------------------- Data structures ---------------------------- #

@dataclass
class Block:
    text: str
    page: int
    x0: float
    x1: float
    top: float
    bottom: float
    font_size: float = 0.0
    bold: bool = False
    ital: bool = False
    type: str = "PARA"  # PARA, HEADING_CAND, CAPTION_CAND, EQUATION, LIST, FOOTNOTE

@dataclass
class Section:
    label: str
    text: str = ""
    units: List[Dict[str, Any]] = field(default_factory=list)

# ---------------------------- Helpers ---------------------------- #

def normalize_heading(h: str) -> str:
    s = h.strip()
    s = HEADING_NUMBER_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def guess_is_heading(line: Block, median_fsize: float) -> bool:
    t = line.text.strip()
    if not t:
        return False
    length_ok = len(t) <= 120
    shortish = len(t.split()) <= 14
    font_bonus = line.font_size >= (median_fsize + 1.5) if line.font_size else False
    bold_bonus = line.bold
    numbered = bool(re.match(r"^(?:[0-9IVX]+[\.)])\s+", t))
    punct_ok = t.endswith(":") or not t.endswith(".")
    return length_ok and shortish and punct_ok and (font_bonus or bold_bonus or numbered)

# ---------------------------- Parsing ---------------------------- #

def parse_pdf_blocks(path: str) -> List[Block]:
    """
    Prefer PyMuPDF dict blocks (stable across versions). Fallback to pdfplumber.
    If neither available and input is .txt, split by blank lines.
    """
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        paras = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
        return [Block(text=p, page=0, x0=0, x1=0, top=i*100, bottom=(i+1)*100) for i, p in enumerate(paras)]

    blocks: List[Block] = []

    # --- Prefer PyMuPDF: dict output ---
    if fitz is not None:
        try:
            doc = fitz.open(path)
            for p_idx, page in enumerate(doc):
                d = page.get_text("dict")  # stable schema
                for b in d.get("blocks", []):
                    if "lines" not in b:
                        continue
                    # Rebuild paragraph text from spans
                    lines_txt = []
                    for line in b["lines"]:
                        spans = line.get("spans", [])
                        seg = "".join(span.get("text", "") for span in spans).strip()
                        if seg:
                            lines_txt.append(seg)
                    t = "\n".join(lines_txt).strip()
                    if not t:
                        continue
                    x0, y0, x1, y1 = b.get("bbox", [0, 0, 0, 0])
                    # Split very large blocks on blank lines to avoid blobs
                    for i, para in enumerate(re.split(r"\n\s*\n", t)):
                        para = para.strip()
                        if not para:
                            continue
                        blocks.append(Block(
                            text=para, page=p_idx, x0=x0, x1=x1,
                            top=y0 + i*12, bottom=y0 + (i+1)*12
                        ))
        except Exception as e:
            print(f"[warn] PyMuPDF failed: {e}", file=sys.stderr)

    # --- Fallback: pdfplumber (lines) ---
    if not blocks and pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                for p_idx, page in enumerate(pdf.pages):
                    try:
                        chars = page.chars
                        if not chars:
                            raise ValueError("no chars")
                    except Exception:
                        text = page.extract_text(x_tolerance=1, y_tolerance=2) or ""
                        for i, para in enumerate(re.split(r"\n\s*\n", text)):
                            para = para.strip()
                            if not para: continue
                            blocks.append(Block(text=para, page=p_idx, x0=0, x1=page.width, top=i*12, bottom=(i+1)*12))
                        continue

                    by_line: Dict[int, List[Dict[str, Any]]] = {}
                    for ch in chars:
                        y = int(round(ch.get("top", 0)))
                        by_line.setdefault(y, []).append(ch)
                    for y, arr in sorted(by_line.items(), key=lambda kv: kv[0]):
                        arr.sort(key=lambda c: c.get("x0", 0))
                        text = "".join(c.get("text", "") for c in arr)
                        if not text.strip():
                            continue
                        x0 = min(c.get("x0", 0) for c in arr)
                        x1 = max(c.get("x1", 0) for c in arr)
                        top = min(c.get("top", 0) for c in arr)
                        bottom = max(c.get("bottom", 0) for c in arr)
                        sizes = [float(c.get("size", 0)) for c in arr if c.get("size")]
                        fsize = sum(sizes)/len(sizes) if sizes else 0.0
                        fontnames = [c.get("fontname", "") for c in arr]
                        bold = any("Bold" in fn or fn.endswith("Bd") for fn in fontnames)
                        ital = any("Italic" in fn or fn.endswith("It") for fn in fontnames)
                        blocks.append(Block(text=text, page=p_idx, x0=x0, x1=x1, top=top, bottom=bottom,
                                            font_size=fsize, bold=bold, ital=ital))
        except Exception as e:
            print(f"[warn] pdfplumber failed: {e}", file=sys.stderr)

    if not blocks:
        raise RuntimeError("No text extracted. Install `pymupdf` or `pdfplumber` and retry.")
    return blocks

# ---------------------------- Headings & Paragraphs ---------------------------- #

INLINE_HEAD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z \-/&]+?)\s*[:–—-]\s+(.*)$")

def detect_heading_candidates(blocks: List[Block]) -> List[Block]:
    fs = [b.font_size for b in blocks if b.font_size > 0]
    median_fsize = sorted(fs)[len(fs)//2] if fs else 10.0

    out: List[Block] = []
    for b in blocks:
        text = b.text.strip()
        lines = text.splitlines() or [text]
        first = lines[0].strip()

        # Case A: "Abstract: ..." / "Keywords: ..." inline style
        m_inline = INLINE_HEAD_RE.match(first)
        if m_inline:
            head = normalize_heading(m_inline.group(1))
            if map_to_ontology(head):
                hb = Block(**{**b.__dict__}); hb.text = head; hb.type = "HEADING_CAND"
                out.append(hb)
                # Keep the rest of the FIRST line + all remaining lines
                remainder_first = m_inline.group(2).strip()
                remainder_text = "\n".join([remainder_first] + lines[1:]).strip()
                if remainder_text:
                    pb = Block(**{**b.__dict__}); pb.text = remainder_text; pb.type = "PARA"
                    out.append(pb)
                continue

        # Case B: standalone heading line (typography or ontology)
        is_head_typo = guess_is_heading(b, median_fsize)
        text_norm = normalize_heading(first)
        textual_heading = (0 < len(text_norm.split()) <= 7) and (map_to_ontology(text_norm) is not None)

        if is_head_typo or textual_heading:
            rest = "\n".join(lines[1:]).strip()
            hb = Block(**{**b.__dict__}); hb.text = text_norm; hb.type = "HEADING_CAND"
            out.append(hb)
            if rest:
                pb = Block(**{**b.__dict__}); pb.text = rest; pb.type = "PARA"
                out.append(pb)
        else:
            if re.match(r"^(Figure|Fig\.|Table)\s*\d+[:\.]", text, re.I):
                b2 = Block(**{**b.__dict__}); b2.type = "CAPTION_CAND"; out.append(b2)
            else:
                b2 = Block(**{**b.__dict__}); b2.type = "PARA"; out.append(b2)
    return out

def map_to_ontology(title: str) -> Optional[str]:
    t = normalize_heading(title).lower()
    base = re.sub(r"[^\w\s]", "", t)
    base = re.sub(r"\s+", " ", base).strip()
    for canon, syns in ONTOLOGY.items():
        for s in syns:
            if base == s or base.startswith(s + " "):
                return canon
    if base in [k.lower() for k in CANON]:
        return base.upper()
    return None

import statistics
BULLET_RE = re.compile(r"^\s*(?:[\u2022\-\u2013\*]|(?:\(?\d+[\)\.])|(?:[A-Za-z]\)))\s+")
SENT_END_RE = re.compile(r'[.!?]["\')\]]?$')

def _likely_cont(prev_text: str, curr_text: str) -> bool:
    pt = prev_text.rstrip()
    ct = curr_text.lstrip()
    if not pt: return False
    if SENT_END_RE.search(pt): return False
    # continuation if next starts with lowercase/number/paren or a comma/semicolon dash
    return bool(re.match(r"^[a-z0-9(\[,:;–-]", ct))

def merge_lines_into_paragraphs(lines: List[Block], base_y_gap: float = 8.0) -> List[Block]:
    if not lines: return []

    by_page: Dict[int, List[Block]] = {}
    for ln in lines: by_page.setdefault(ln.page, []).append(ln)
    page_gap: Dict[int, float] = {}
    for pg, arr in by_page.items():
        heights = [max(1.0, b.bottom - b.top) for b in arr]
        med_h = statistics.median(heights) if heights else 12.0
        page_gap[pg] = max(base_y_gap, 1.9 * med_h)  # a bit more forgiving

    paras: List[Block] = []
    curr: Optional[Block] = None
    last_bottom = last_x0 = last_x1 = None
    in_bullet = False
    bullet_indent = None

    # helper to append text with soft hyphen repair
    def append_text(dst: Block, src: Block):
        sep = " " if not dst.text.endswith("-") else ""
        dst.text = dst.text.rstrip("-") + sep + src.text.strip()
        dst.bottom = src.bottom

    sorted_lines = sorted(lines, key=lambda b: (b.page, b.top, b.x0))
    for ln in sorted_lines:
        y_gap_threshold = page_gap.get(ln.page, base_y_gap)
        is_bullet = bool(BULLET_RE.match(ln.text))

        if curr is None:
            curr = Block(**{**ln.__dict__})
            last_bottom, last_x0, last_x1 = ln.bottom, ln.x0, ln.x1
            in_bullet = is_bullet
            if is_bullet:
                m = BULLET_RE.match(ln.text)
                bullet_indent = (ln.x0 + (m.end() - m.start()) * 3) if m else (ln.x0 + 12)
            else:
                bullet_indent = None
            continue

        same_col_loose = (last_x0 is not None and abs(ln.x0 - last_x0) < 60 and abs(ln.x1 - last_x1) < 80)
        gap_big = (ln.top - (last_bottom or ln.top)) > y_gap_threshold
        new_para = (ln.page != curr.page) or gap_big or not same_col_loose

        # Bullet continuation: same item if indented further and not a fresh bullet
        if in_bullet:
            if not is_bullet and bullet_indent is not None and ln.x0 > (bullet_indent + 4):
                append_text(curr, ln)
                last_bottom, last_x0, last_x1 = ln.bottom, ln.x0, ln.x1
                continue
            # New bullet in same list
            if is_bullet and same_col_loose and not gap_big:
                paras.append(curr)
                curr = Block(**{**ln.__dict__})
                m = BULLET_RE.match(ln.text)
                bullet_indent = (ln.x0 + (m.end() - m.start()) * 3) if m else (ln.x0 + 12)
                last_bottom, last_x0, last_x1 = ln.bottom, ln.x0, ln.x1
                continue
            # Bullet list ended → fall through
            in_bullet = False
            bullet_indent = None

        # Wrapped-line continuation (even across mild col shifts/page breaks)
        if new_para and _likely_cont(curr.text, ln.text):
            # permit bridging across page break or small column drift
            if (ln.page == curr.page) or (ln.page == curr.page + 1):
                append_text(curr, ln)
                last_bottom, last_x0, last_x1 = ln.bottom, ln.x0, ln.x1
                continue

        if new_para:
            paras.append(curr)
            curr = Block(**{**ln.__dict__})
            in_bullet = is_bullet
            if in_bullet:
                m = BULLET_RE.match(ln.text)
                bullet_indent = (ln.x0 + (m.end() - m.start()) * 3) if m else (ln.x0 + 12)
            else:
                bullet_indent = None
        else:
            append_text(curr, ln)

        last_bottom, last_x0, last_x1 = ln.bottom, ln.x0, ln.x1

    if curr is not None:
        paras.append(curr)
    return paras

def normalize_keywords_in_sections(sections: List[Section]) -> None:
    for s in sections:
        if s.label == "KEYWORDS":
            raw = " ".join(u.get("text","") for u in s.units)
            raw = re.sub(r'^\s*keywords?\s*[:\-–—]?\s*', '', raw, flags=re.I)
            raw = re.sub(r'\s+', ' ', raw).strip()
            tokens = [t.strip(" .;:") for t in re.split(r'[;,]', raw) if t.strip()]
            # Ensure clean text and expose tokens
            s.text = "Keywords: " + "; ".join(tokens)
            s.units = [{"type": "KEYWORD", "text": t} for t in tokens]

# ---------------------------- Section assignment ---------------------------- #

def assign_sections_by_headings(paras: List[Block]) -> Tuple[List[Tuple[str, Block]], float]:
    """Greedy assignment: each paragraph inherits the most recent mapped heading."""
    hmap: List[Tuple[int, str]] = []
    for i, p in enumerate(paras):
        if p.type == "HEADING_CAND":
            lab = map_to_ontology(p.text)
            if lab:
                hmap.append((i, lab))
    if not hmap:
        return [("UNKNOWN", p) for p in paras], 0.0

    # Rough confidence = coverage of headings with a tiny order sanity penalty
    labels_in_order = [lab for _, lab in hmap]
    order_penalty = sum(1 for a, b in zip(labels_in_order, labels_in_order[1:])
                        if a == "RESULTS" and b in {"METHODS", "INTRODUCTION"})
    coverage = min(1.0, len(hmap) / max(1, len(paras)/6))
    heading_conf = max(0.0, min(1.0, coverage - 0.1*order_penalty))

    labeled: List[Tuple[str, Block]] = []
    curr_label = "UNKNOWN"
    h_idx = 0
    for i, p in enumerate(paras):
        while h_idx < len(hmap) and hmap[h_idx][0] <= i:
            curr_label = hmap[h_idx][1]
            h_idx += 1
        labeled.append((curr_label, p))
    return labeled, heading_conf

# ---------------------------- Grouping & Fallback ---------------------------- #

def build_sections(labeled_paras: List[Tuple[str, Block]]) -> List[Section]:
    out: List[Section] = []
    curr: Optional[Section] = None
    last_label: Optional[str] = None
    for lab, p in labeled_paras:
        if lab != last_label:
            if curr is not None:
                curr.text = "\n\n".join(u["text"] for u in curr.units if u["type"] == "PARA")
                out.append(curr)
            curr = Section(label=lab, units=[])
            last_label = lab
        curr.units.append({
            "type": "PARA",
            "text": p.text,
            "page": p.page,
            "x0": getattr(p, "x0", 0.0),
            "x1": getattr(p, "x1", 0.0),
            "top": getattr(p, "top", 0.0),
            "bottom": getattr(p, "bottom", 0.0),
        })
    if curr is not None:
        curr.text = "\n\n".join(u["text"] for u in curr.units if u["type"] == "PARA")
        out.append(curr)
    return out

# -------- Table repair v2 (handles "Title & Author(s), Year / Methodology / Key Findings / Limitations") ----
import math

# Fuzzy header patterns for this paper/journal
_HDR_PATTERNS = [
    ("SN", re.compile(r"^\s*S/?N\b", re.I)),
    ("TITLE_YEAR", re.compile(r"title\s*&?\s*author\(s\)[, ]*\s*year", re.I)),
    ("METHODOLOGY", re.compile(r"^methodolog(y|ies)\b", re.I)),
    ("KEY_FINDINGS", re.compile(r"^key\s*findings\b", re.I)),
    ("LIMITATIONS", re.compile(r"^limitations\b", re.I)),
    ("OBSERVATIONS", re.compile(r"^observations\b", re.I)),
    # fallbacks seen in your JSON
    ("DATASET", re.compile(r"^dataset\b", re.I)),
    ("CONTRIB", re.compile(r"^contrib(ution)?s?\b", re.I)),
    ("OBS", re.compile(r"^obs(ervations)?\b", re.I)),
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _join_soft(a: str, b: str) -> str:
    if not a: return b.strip()
    a = a.rstrip("-")
    return (a + (" " if not a.endswith("-") else "") + b.strip())

def _within(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol

def _cluster_by_y(items, y_tol):
    """Group items into visual rows using y-center clustering."""
    rows = []
    for u in sorted(items, key=lambda u: (u["page"], (u.get("top",0)+u.get("bottom",0))/2.0, u.get("x0",0))):
        yc = (float(u.get("top",0.0)) + float(u.get("bottom",0.0))) / 2.0
        placed = False
        for r in rows:
            if _within(r["yc"], yc, y_tol) and u["page"] == r["page"]:
                r["yc"] = (r["yc"]*len(r["items"]) + yc) / (len(r["items"])+1)
                r["items"].append(u)
                placed = True
                break
        if not placed:
            rows.append({"page": u["page"], "yc": yc, "items": [u]})
    return rows

def _detect_header_band(rows):
    """
    Find a row (or 2-3 stacked rows) containing most header tokens.
    Returns (header_units, header_row_index_start, header_row_index_end)
    """
    best = None
    for i in range(len(rows)):
        for j in range(i, min(i+3, len(rows))):
            band = []
            for k in range(i, j+1):
                band.extend(rows[k]["items"])
            labels = []
            for u in band:
                txt = _norm(u.get("text","")).lower()
                for name, rx in _HDR_PATTERNS:
                    if rx.search(txt):
                        labels.append((name, u))
                        break
            # need at least two of the core headers
            core = {"TITLE_YEAR","METHODOLOGY","KEY_FINDINGS"} & {n for n,_ in labels}
            if len(core) >= 2:
                best = (band, i, j)
                break
        if best: break
    return best

def _build_columns_from_header(header_units):
    """
    Build ordered columns (name, x0) from header band.
    Joins wrapped header chunks like 'Title &' + 'Author(s), Year'.
    """
    # First, combine text by x position buckets
    units = sorted(header_units, key=lambda u: float(u.get("x0",0.0)))
    buckets = []  # list of dict{x0, texts[]}
    for u in units:
        x = float(u.get("x0",0.0))
        placed = False
        for b in buckets:
            if abs(b["x0"] - x) <= 30:  # merge close header chunks
                b["x0"] = (b["x0"]*len(b["texts"]) + x) / (len(b["texts"])+1)
                b["texts"].append(_norm(u.get("text","")))
                placed = True
                break
        if not placed:
            buckets.append({"x0": x, "texts": [_norm(u.get("text",""))]})

    cols = []
    for b in buckets:
        text = _norm(" ".join(b["texts"]))
        name = None
        for n, rx in _HDR_PATTERNS:
            if rx.search(text.lower()):
                name = n; break
        # Heuristic join for TITLE & Author(s), Year
        if not name and "title" in text.lower() and "author" in text.lower():
            name = "TITLE_YEAR"
        if name:
            cols.append((name, b["x0"]))

    # Keep only one of LIMITATIONS/OBSERVATIONS/OBS — treat all as LIMITATIONS
    normalized = []
    seen = set()
    for n,x in cols:
        if n in {"OBSERVATIONS","OBS"}: n = "LIMITATIONS"
        if n not in seen:
            normalized.append((n,x)); seen.add(n)
    # Ensure order left->right
    normalized.sort(key=lambda t: t[1])
    # Drop S/N if present; it is optional and often noisy
    normalized = [(n,x) for (n,x) in normalized if n != "SN"]
    return normalized

def _assign_col_idx(x0, cols):
    best, bestd = 0, float("inf")
    for i, (_, cx) in enumerate(cols):
        d = abs(x0 - cx)
        if d < bestd:
            best, bestd = i, d
    return best

def _consume_table(rows, start_row, cols):
    """
    From start_row+1 onward, collect lines into table rows until a gap/heading break.
    Returns (table_dict, end_row_index_exclusive)
    """
    if len(cols) < 2:
        return None, start_row+1

    # Approximate line height
    heights = []
    for r in rows:
        for u in r["items"]:
            h = float(u.get("bottom",0.0)) - float(u.get("top",0.0))
            if h > 0: heights.append(h)
    line_h = (sum(heights)/len(heights)) if heights else 12.0
    y_tol = max(10.0, 1.6*line_h)

    data_rows = []
    cur_y = None
    cur_cells = [""] * len(cols)

    def flush():
        nonlocal cur_cells
        if any(_norm(c) for c in cur_cells):
            data_rows.append([_norm(c) for c in cur_cells])
        cur_cells = [""] * len(cols)

    # Walk subsequent rows, same page and next pages until a new section-like heading appears
    r = start_row + 1
    last_page = rows[start_row]["page"]
    while r < len(rows):
        page = rows[r]["page"]
        # Break if we hit a likely new narrative block: a single wide item spanning ~full width
        items = rows[r]["items"]
        if len(items) == 1:
            span = float(items[0].get("x1",0.0)) - float(items[0].get("x0",0.0))
            if span > 400:  # page width ~ 540; tune if needed
                break

        for u in items:
            if u.get("type") != "PARA": 
                continue
            x = float(u.get("x0",0.0))
            y = (float(u.get("top",0.0)) + float(u.get("bottom",0.0))) / 2.0
            col = _assign_col_idx(x, cols)

            if cur_y is None:
                cur_y = y

            if (page != last_page) or (abs(y - cur_y) > y_tol):
                flush()
                cur_y = y
                last_page = page

            cur_cells[col] = _join_soft(cur_cells[col], u.get("text",""))

        r += 1

    flush()

    headers = [n for (n, _) in cols]
    # map header canonicalization
    header_map = {
        "TITLE_YEAR": "TITLE_AUTHOR_YEAR",
        "METHODOLOGY": "METHODOLOGY",
        "KEY_FINDINGS": "KEY_FINDINGS",
        "LIMITATIONS": "LIMITATIONS",
        "DATASET": "DATASET",
        "CONTRIB": "CONTRIB",
    }
    headers_std = [header_map.get(h, h) for h in headers]

    table = {
        "type": "TABLE",
        "headers": headers_std,
        "rows": [{headers_std[i]: cell for i, cell in enumerate(row)} for row in data_rows]
    }
    return table, r

def repair_tables_in_sections_v2(sections: List[Section]) -> int:
    """
    Robust repair for this journal's comparison table.
    Detect header band, build columns, and reconstruct rows across following pages.
    """
    fixed = 0
    for s in sections:
        # Work per page: build visual rows with coordinates preserved
        units = [u for u in s.units if u.get("type") in {"PARA"}]
        if not units:
            continue

        # Cluster lines into rows per page
        # Estimate y_tol from median height
        heights = [max(1.0, float(u.get("bottom",0.0)) - float(u.get("top",0.0))) for u in units]
        line_h = (sum(heights)/len(heights)) / max(1, len(heights)) if heights else 12.0
        y_tol = max(10.0, 1.8 * (line_h if line_h else 12.0))

        rows = _cluster_by_y(units, y_tol)
        if not rows:
            continue

        header_band = _detect_header_band(rows)
        if not header_band:
            continue
        header_units, i0, j0 = header_band
        cols = _build_columns_from_header(header_units)
        if len(cols) < 2:
            continue

        table, end_row = _consume_table(rows, j0, cols)
        if not table or not table.get("rows"):
            continue

        # Replace the units from header band start to end_row with the table
        # Find the index range in s.units that corresponds to those row items
        band_item_ids = set(id(u) for k in range(i0, end_row) for u in rows[k]["items"])
        new_units = []
        replaced = False
        for u in s.units:
            if id(u) in band_item_ids:
                if not replaced:
                    new_units.append(table)
                    replaced = True
                # skip originals
            else:
                new_units.append(u)

        if replaced:
            s.units = new_units
            # recompute text without table blob
            s.text = "\n\n".join(u["text"] for u in s.units if u.get("type") == "PARA")
            fixed += 1

    return fixed

# ---------------------------- Pipeline ---------------------------- #

def process_document(path: str, min_conf: float = 0.4) -> Dict[str, Any]:
    raw_blocks = parse_pdf_blocks(path)
    blocks = detect_heading_candidates(raw_blocks)
    paras = merge_lines_into_paragraphs(blocks)
    labeled_by_head, head_conf = assign_sections_by_headings(paras)
    mode = "L3" if head_conf >= min_conf else "L0"
    sections = build_sections(labeled_by_head)
    # If you already added the keywords normalizer, keep that call first.
    try:
        normalize_keywords_in_sections(sections)  # keep if you added this earlier
    except Exception:
        pass

    tables_fixed = repair_tables_in_sections_v2(sections)

    normalize_keywords_in_sections(sections)
    return {
        "mode": mode,
        "sections": [s.__dict__ for s in sections],
        "telemetry": {
            "heading_confidence": head_conf,
            "num_pages": max((b.page for b in raw_blocks), default=0) + 1,
            "num_sections": len(sections),
            "tables_fixed": tables_fixed,
        },
    }

# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Format-agnostic scientific sectioner (classifier-free)")
    ap.add_argument("input", help="PDF or .txt file")
    ap.add_argument("--out", default="sections.json", help="Output JSON path")
    ap.add_argument("--min-conf", type=float, default=0.4, help="Min heading confidence to accept (default 0.4)")
    args = ap.parse_args()

    result = process_document(args.input, min_conf=args.min_conf)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}. Mode={result['mode']} Sections={len(result['sections'])}")

if __name__ == "__main__":
    main()
