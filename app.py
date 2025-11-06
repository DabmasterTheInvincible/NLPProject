from __future__ import annotations

import base64
import json
import os
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from io import BytesIO

try:
    import fitz
except Exception:
    fitz = None

import processing
from processing import DocumentData, DocumentProcessor
from stylometry import features_for_windows as stylometry_features
from perplexity import PerplexityEntropyAnalyzer
from semantic import E5Embedder, SemanticIndex, search_windows_for_paraphrase
from deberta import load as load_deberta, predict_proba as deberta_predict_proba

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WINDOW_SIZE = 3
WINDOW_STRIDE = 2
MIN_SENTENCE_CHARS = 15
MAX_WINDOW_TOKENS = 512
DEFAULT_TAU = float(os.environ.get("TAU_SEM", "0.78"))
DEFAULT_MAX_PAGES = int(os.environ.get("MAX_PDF_PAGES", "40"))

DEBERTA_DIR = os.environ.get("DEBERTA_DIR", "./models/deberta")
META_CKPT = os.environ.get("META_CKPT", "./models/meta_clf/meta_clf.pth")
META_COLS = os.environ.get("META_COLS", "./models/meta_clf/meta_clf.columns.json")
AI_INDEX_DIR = os.environ.get("AI_INDEX_DIR", "./indices/ai_corpus_index")
EMB_MODEL = os.environ.get("EMB_MODEL", "intfloat/e5-small-v2")

DEFAULT_LABEL_ORDER = ["human", "ai", "paraphrased_ai"]
PROB_COLUMNS = [f"p_{name}" for name in DEFAULT_LABEL_ORDER]
COLOR_MAP = {"human": "#22c55e", "ai": "#ef4444", "paraphrased_ai": "#f59e0b"}
CONFIDENCE_THRESHOLD = 0.6
DETECTGPT_THRESHOLD = -0.02
COVERAGE_AI_THRESHOLD = 0.35
COVERAGE_PARA_THRESHOLD = 0.3


@dataclass
class PipelineConfig:
    tau_semantic: float = DEFAULT_TAU
    detectgpt_threshold: float = DETECTGPT_THRESHOLD
    coverage_ai_threshold: float = COVERAGE_AI_THRESHOLD
    coverage_para_threshold: float = COVERAGE_PARA_THRESHOLD
    confidence_threshold: float = CONFIDENCE_THRESHOLD


@dataclass
class AnalysisResult:
    document: DocumentData
    window_df: pd.DataFrame
    sentence_df: pd.DataFrame
    section_summaries: List[Dict[str, Any]]
    document_summary: Dict[str, Any]
    alerts: List[str]
    highlight_html: str
    export_payload: Dict[str, Any]


# -----------------------------------------------------------------------------
# Cached resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_processor() -> DocumentProcessor:
    return DocumentProcessor(
        sentence_splitter="regex",
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        min_sentence_chars=MIN_SENTENCE_CHARS,
    )


@st.cache_resource(show_spinner=False)
def get_perplexity_analyzer() -> Optional[PerplexityEntropyAnalyzer]:
    try:
        return PerplexityEntropyAnalyzer(
            model_name="gpt2-medium",
            max_len=MAX_WINDOW_TOKENS,
            batch_size=4,
            top_k_entropy=100,
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_semantic_resources(index_dir: str, embed_model: str) -> Tuple[Optional[E5Embedder], Optional[SemanticIndex]]:
    if not os.path.isdir(index_dir):
        return None, None
    try:
        embedder = E5Embedder(embed_model, max_len=256, batch_size=32)
        index = SemanticIndex.load(index_dir)
        return embedder, index
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def load_meta_resources(ckpt_path: str, cols_path: str):
    if not (os.path.isfile(ckpt_path) and os.path.isfile(cols_path)):
        return None
    try:
        model = torch.load(ckpt_path, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()
    except Exception:
        return None
    try:
        with open(cols_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if isinstance(meta, list):
        feat_cols = meta
        mu = np.zeros(len(feat_cols), dtype=np.float32)
        sigma = np.ones(len(feat_cols), dtype=np.float32)
        label_map = {0: "human", 1: "ai", 2: "paraphrased_ai"}
    else:
        feat_cols = meta.get("columns") or meta.get("feature_names") or meta.get("cols") or []
        mu = np.array(meta.get("mu") or meta.get("mean") or [0.0] * len(feat_cols), dtype=np.float32)
        sigma = np.array(meta.get("sigma") or meta.get("std") or [1.0] * len(feat_cols), dtype=np.float32)
        label_map = meta.get("label_map") or {0: "human", 1: "ai", 2: "paraphrased_ai"}
        label_map = {int(k): v for k, v in label_map.items()}
    return model, feat_cols, mu, sigma, label_map


# -----------------------------------------------------------------------------
# Helper utilities

def compute_meta_window_probs(window_df: pd.DataFrame, meta_bundle, sem_doc: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not meta_bundle:
        return None
    model, feat_cols, mu, sigma, label_map = meta_bundle
    if model is None or not feat_cols:
        return None
    features = pd.DataFrame(index=window_df.index)
    for col in feat_cols:
        if col in window_df.columns:
            features[col] = window_df[col]
        elif col in {"coverage", "max_sim", "top3_mean", "frac_above_tau"}:
            val = 0.0
            if isinstance(sem_doc, dict):
                val = sem_doc.get(col, 0.0)
            features[col] = val
        elif col == "num_windows":
            features[col] = float(len(window_df))
        else:
            features[col] = 0.0
    features = features.fillna(0.0)
    X = features[feat_cols].to_numpy(dtype=np.float32)
    if mu is not None and sigma is not None and len(mu) == X.shape[1]:
        mu_arr = np.asarray(mu, dtype=np.float32)
        sigma_arr = np.asarray(sigma, dtype=np.float32)
        sigma_arr[sigma_arr == 0] = 1.0
        X = (X - mu_arr) / sigma_arr
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    label_order = [normalize_label_name(label_map[idx]) for idx in sorted(label_map.keys())]
    out = pd.DataFrame(0.0, index=window_df.index, columns=PROB_COLUMNS)
    for idx, name in enumerate(label_order):
        col = f"p_{name}"
        if col in out.columns:
            out[col] = probs[:, idx]
    return out


HIGHLIGHT_RGB = {
    "human": (34 / 255, 197 / 255, 94 / 255),
    "ai": (239 / 255, 68 / 255, 68 / 255),
    "paraphrased_ai": (245 / 255, 158 / 255, 11 / 255),
}

PIE_COLORS = {
    "human": "#22c55e",
    "ai": "#ef4444",
    "paraphrased_ai": "#f59e0b",
}

def generate_highlighted_pdf(pdf_bytes: bytes, sentence_df: pd.DataFrame) -> Optional[bytes]:
    if fitz is None or pdf_bytes is None or sentence_df.empty:
        return None
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None
    try:
        first_page = pdf[0] if len(pdf) else None
        if first_page is not None:
            width, height = first_page.rect.width, first_page.rect.height
        else:
            width, height = 612, 792
        flags = 0
        if hasattr(fitz, "TEXT_DEHYPHENATE"):
            flags |= fitz.TEXT_DEHYPHENATE
        if hasattr(fitz, "TEXT_IGNORECASE"):
            flags |= fitz.TEXT_IGNORECASE
        word_cache: Dict[int, List[Tuple[float, float, float, float, str]]] = {}

        def ensure_words(page):
            idx = page.number
            if idx not in word_cache:
                try:
                    word_cache[idx] = page.get_text("words") or []
                except Exception:
                    word_cache[idx] = []
            return word_cache[idx]

        for row in sentence_df.itertuples():
            page_no = getattr(row, "page", None)
            if page_no is None or (isinstance(page_no, float) and np.isnan(page_no)):
                continue
            text = getattr(row, "text", "") or ""
            clean = getattr(row, "clean_text", "") or ""
            try:
                page = pdf[int(page_no) - 1] if page_no > 0 else pdf[int(page_no)]
            except Exception:
                continue
            areas: List[Any] = []
            normalized_clean = " ".join(clean.split())
            normalized_text = " ".join(text.split())
            snippets = []
            for base in (normalized_text, normalized_clean):
                if not base:
                    continue
                variants = [base, base.replace("- ", "-"), base.replace(" -", "-")]
                words = base.split()
                max_n = min(len(words), 8)
                min_n = 3 if len(words) > 3 else len(words)
                for variant in variants:
                    if variant:
                        snippets.append(variant)
                for n in range(max_n, min_n - 1, -1):
                    snippet = " ".join(words[:n]).strip()
                    if snippet:
                        snippets.append(snippet)
            seen = set()
            for snippet in snippets:
                key = snippet.lower()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    hits = page.search_for(snippet, hit_max=64, quads=True, flags=flags)
                except Exception:
                    hits = []
                if hits:
                    areas = hits
                    break
            if not areas:
                words = ensure_words(page)
                tokens = [w[4] for w in words]
                lowered = [tok.lower() for tok in tokens]
                target = normalized_clean.lower() or normalized_text.lower()
                target_tokens = target.split()
                if words and target_tokens:
                    win = len(target_tokens)
                    for i in range(0, len(words) - win + 1):
                        if lowered[i:i + win] == target_tokens:
                            rect = fitz.Rect(words[i][0], words[i][1], words[i][2], words[i][3])
                            for j in range(i + 1, i + win):
                                rect |= fitz.Rect(words[j][0], words[j][1], words[j][2], words[j][3])
                            areas = [rect]
                            break
            if not areas:
                try:
                    blocks = page.get_text("blocks") or []
                except Exception:
                    blocks = []
                target_block = normalized_clean.lower() or normalized_text.lower()
                for bx in blocks:
                    rect = fitz.Rect(bx[:4])
                    block_text = " ".join((bx[4] or "").split()).lower()
                    if target_block and target_block in block_text:
                        areas = [rect]
                        break
            if not areas:
                bbox_vals = [getattr(row, name, None) for name in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")]
                if all(val is not None and not (isinstance(val, float) and math.isnan(val)) for val in bbox_vals):
                    try:
                        rect = fitz.Rect(float(bbox_vals[0]), float(bbox_vals[1]), float(bbox_vals[2]), float(bbox_vals[3]))
                        areas = [rect]
                    except Exception:
                        areas = []
            if not areas:
                continue
            color = HIGHLIGHT_RGB.get(getattr(row, "label", "human"), HIGHLIGHT_RGB["human"])
            for area in areas:
                try:
                    target_rect = area if isinstance(area, fitz.Rect) else fitz.Rect(area)
                    annot = page.add_highlight_annot(target_rect)
                    annot.set_colors(stroke=color)
                    annot.set_opacity(0.4)
                    annot.update()
                except Exception:
                    continue
        legend_page = pdf.new_page(pno=0, width=width, height=height)
        legend_page.insert_text((72, 72), "AI Detection Legend", fontsize=24, color=(0, 0, 0))
        distribution = sentence_df['label'].value_counts(normalize=True) if 'label' in sentence_df.columns else {}
        y = 120
        for label in DEFAULT_LABEL_ORDER:
            color = HIGHLIGHT_RGB[label]
            rect = fitz.Rect(72, y - 12, 112, y + 12)
            legend_page.draw_rect(rect, color=color, fill=color, overlay=True)
            pct = float(distribution.get(label, 0.0) * 100.0)
            legend_page.insert_text((120, y + 5), f"{label.replace('_', ' ').title()} ({pct:.1f}%)", fontsize=14, color=(0, 0, 0))
            y += 36
        legend_page.insert_text((72, y + 10), "Highlighted spans indicate predicted authorship.", fontsize=12)
        buffer = BytesIO()
        pdf.save(buffer)
        buffer.seek(0)
        return buffer.read()
    finally:
        try:
            pdf.close()
        except Exception:
            pass

def build_pie_chart_html(percentages: Dict[str, float]) -> str:
    total = sum(percentages.values()) or 1.0
    stops = []
    cumulative = 0.0
    for label in DEFAULT_LABEL_ORDER:
        value = percentages.get(label, 0.0)
        share = max(value / total, 0.0) * 100.0
        next_val = cumulative + share
        color = PIE_COLORS[label]
        stops.append(f"{color} {cumulative:.2f}% {next_val:.2f}%")
        cumulative = next_val
    gradient = ", ".join(stops) or "#cccccc 0% 100%"
    legend_items = []
    for label in DEFAULT_LABEL_ORDER:
        legend_items.append(
            f"<div class='legend-item'><span class='legend-swatch' style='background:{PIE_COLORS[label]}'></span>"
            f"{label.replace('_', ' ').title()} ({percentages.get(label, 0.0):.1f}%)</div>"
        )
    css = (
        "<style>"
        ".pie-wrapper{display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;}"
        ".pie-chart{width:180px;height:180px;border-radius:50%;box-shadow:0 0 10px rgba(0,0,0,0.1);}"
        ".legend{display:flex;flex-direction:column;gap:0.4rem;font-size:0.9rem;}"
        ".legend-item{display:flex;align-items:center;}"
        ".legend-swatch{display:inline-block;width:16px;height:16px;border-radius:3px;margin-right:0.5rem;}"
        "</style>"
    )
    return css + (
        "<div class='pie-wrapper'>"
        f"<div class='pie-chart' style='background: conic-gradient({gradient});'></div>"
        f"<div class='legend'>{''.join(legend_items)}</div>"
        "</div>"
    )


def build_color_legend_html() -> str:
    items = []
    for label in DEFAULT_LABEL_ORDER:
        items.append(
            f"<div class='legend-item'><span class='legend-swatch' style='background:{PIE_COLORS[label]}'></span>"
            f"{label.replace('_', ' ').title()}</div>"
        )
    css = (
        "<style>"
        ".inline-legend{display:flex;gap:1rem;flex-wrap:wrap;}"
        ".legend-item{display:flex;align-items:center;font-size:0.9rem;}"
        ".legend-swatch{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:0.4rem;}"
        "</style>"
    )
    return css + f"<div class='inline-legend'>{''.join(items)}</div>"


def normalize_label_name(label: str) -> str:
    name = (label or "").lower().strip()
    mapping = {
        "ai_generated": "ai",
        "machine": "ai",
        "paraphrased": "paraphrased_ai",
        "paraphrase": "paraphrased_ai",
        "rewritten": "paraphrased_ai",
    }
    return mapping.get(name, name)


def compute_deberta_probs(window_texts: List[str], window_ids: List[int], alerts: List[str]) -> pd.DataFrame:
    num = len(window_texts)
    df = pd.DataFrame({"window_id": window_ids})
    if num == 0:
        for col in PROB_COLUMNS:
            df[col] = []
        return df
    try:
        _, _, id2label = load_deberta(DEBERTA_DIR)
        probs = deberta_predict_proba(window_texts, max_length=MAX_WINDOW_TOKENS, batch_size=8, model_dir=DEBERTA_DIR)
        label_order = [normalize_label_name(id2label[idx]) for idx in sorted(id2label.keys())]
    except Exception:
        alerts.append("DeBERTa classifier missing - defaulting to uniform probabilities.")
        probs = np.ones((num, len(DEFAULT_LABEL_ORDER)), dtype=np.float32) / len(DEFAULT_LABEL_ORDER)
        label_order = DEFAULT_LABEL_ORDER
    column_map = {"human": "p_human", "ai": "p_ai", "paraphrased_ai": "p_paraphrased_ai"}
    for col in PROB_COLUMNS:
        df[col] = 0.0
    for idx, name in enumerate(label_order):
        col = column_map.get(name)
        if col:
            df[col] = probs[:, idx]
    summed = df[PROB_COLUMNS].sum(axis=1)
    summed[summed == 0] = 1.0
    df[PROB_COLUMNS] = df[PROB_COLUMNS].div(summed, axis=0)
    return df


def compute_stylometry(window_texts: List[str], window_ids: List[int], alerts: List[str]) -> pd.DataFrame:
    try:
        feat_df = stylometry_features(window_texts, window_ids=window_ids)
        return feat_df
    except Exception:
        alerts.append("Stylometry module failed - features zeroed.")
        return pd.DataFrame({"window_id": window_ids})


def compute_perplexity(window_texts: List[str], window_ids: List[int], analyzer: Optional[PerplexityEntropyAnalyzer], alerts: List[str]) -> pd.DataFrame:
    if analyzer is None:
        alerts.append("Perplexity analyzer not available - using zeros.")
        return pd.DataFrame({
            "window_id": window_ids,
            "ppl": np.zeros(len(window_ids)),
            "mean_entropy": np.zeros(len(window_ids)),
            "surprisal_mean": np.zeros(len(window_ids)),
            "surprisal_var": np.zeros(len(window_ids)),
            "avg_rank": np.zeros(len(window_ids)),
            "detectgpt_score": np.zeros(len(window_ids)),
        })
    try:
        ppl_df = analyzer.features(window_texts, K_detectgpt=5, show_progress=False)
        ppl_df.insert(0, "window_id", window_ids)
        return ppl_df
    except Exception:
        alerts.append("Perplexity scoring failed - using zeros.")
        return pd.DataFrame({
            "window_id": window_ids,
            "ppl": np.zeros(len(window_ids)),
            "mean_entropy": np.zeros(len(window_ids)),
            "surprisal_mean": np.zeros(len(window_ids)),
            "surprisal_var": np.zeros(len(window_ids)),
            "avg_rank": np.zeros(len(window_ids)),
            "detectgpt_score": np.zeros(len(window_ids)),
        })


def compute_semantic(window_texts: List[str], window_ids: List[int], resources: Tuple[Optional[E5Embedder], Optional[SemanticIndex]], tau: float, alerts: List[str]):
    embedder, index = resources
    if embedder is None or index is None:
        alerts.append("Semantic paraphrase index unavailable - similarity features set to zero.")
        df = pd.DataFrame({
            "window_id": window_ids,
            "max_sim": np.zeros(len(window_ids)),
            "mean_top3": np.zeros(len(window_ids)),
            "above_tau": np.zeros(len(window_ids), dtype=np.int32),
            "nn_meta": [None] * len(window_ids),
        })
        doc_features = {"coverage": 0.0, "max_sim": 0.0, "top3_mean": 0.0, "frac_above_tau": 0.0, "num_windows": len(window_ids)}
        return df, doc_features
    try:
        sem_df, doc_features = search_windows_for_paraphrase(
            window_texts,
            index_dir=AI_INDEX_DIR,
            embedder_model=EMB_MODEL,
            max_len=256,
            batch_size=64,
            topk=5,
            tau=tau,
            embedder=embedder,
            index=index,
        )
        sem_df = sem_df.rename(columns={"window_index": "window_id"})
        return sem_df, doc_features
    except Exception:
        alerts.append("Semantic similarity scoring failed - similarity features set to zero.")
        df = pd.DataFrame({
            "window_id": window_ids,
            "max_sim": np.zeros(len(window_ids)),
            "mean_top3": np.zeros(len(window_ids)),
            "above_tau": np.zeros(len(window_ids), dtype=np.int32),
            "nn_meta": [None] * len(window_ids),
        })
        doc_features = {"coverage": 0.0, "max_sim": 0.0, "top3_mean": 0.0, "frac_above_tau": 0.0, "num_windows": len(window_ids)}
        return df, doc_features


def smooth_probabilities(prob_matrix: np.ndarray) -> np.ndarray:
    if prob_matrix.size == 0:
        return prob_matrix
    smoothed = np.zeros_like(prob_matrix)
    n = len(prob_matrix)
    for i in range(n):
        start = max(0, i - 1)
        end = min(n, i + 2)
        smoothed[i] = prob_matrix[start:end].mean(axis=0)
    return smoothed


def build_sentence_df(doc: DocumentData) -> pd.DataFrame:
    rows = []
    for sent in doc.sentences:
        row = {
            "sentence_id": sent.sentence_id,
            "section_id": sent.section_id,
            "section_name": sent.section_name,
            "section_offset": sent.section_offset,
            "text": sent.text,
            "clean_text": sent.clean_text,
            "block_type": sent.block_type,
            "block_id": getattr(sent, "block_id", -1),
            "page": sent.page,
        }
        block_idx = row["block_id"]
        if 0 <= block_idx < len(doc.blocks):
            block = doc.blocks[block_idx]
            metadata = block.metadata or {}
            for key, tgt in (("x0", "bbox_x0"), ("top", "bbox_y0"), ("x1", "bbox_x1"), ("bottom", "bbox_y1")):
                val = metadata.get(key)
                if val is None:
                    continue
                try:
                    row[tgt] = float(val)
                except (TypeError, ValueError):
                    continue
            if "placeholder" in metadata:
                row["placeholder"] = metadata.get("placeholder")
            if "source" in metadata:
                row["block_source"] = metadata.get("source")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_sentence_probabilities(sentence_df: pd.DataFrame, window_df: pd.DataFrame) -> pd.DataFrame:
    if sentence_df.empty:
        return sentence_df
    probs = np.zeros((len(sentence_df), len(PROB_COLUMNS)), dtype=np.float32)
    counts = np.zeros(len(sentence_df), dtype=np.int32)
    if not window_df.empty:
        for row in window_df.itertuples():
            indices = getattr(row, "sentence_indices", []) or []
            if not indices:
                continue
            window_probs = np.array([getattr(row, col) for col in PROB_COLUMNS], dtype=np.float32)
            for sid in indices:
                if 0 <= sid < len(sentence_df):
                    probs[sid] += window_probs
                    counts[sid] += 1
    mask = counts > 0
    if mask.any():
        probs[mask] = probs[mask] / counts[mask, None]
    if (~mask).any():
        probs[~mask, 0] = 1.0  # default to human
    sentence_df = sentence_df.copy()
    for idx, col in enumerate(PROB_COLUMNS):
        sentence_df[col] = probs[:, idx]
    labels = np.array(DEFAULT_LABEL_ORDER)[probs.argmax(axis=1)]
    confidence = probs.max(axis=1)
    sentence_df["label"] = labels
    sentence_df["confidence"] = confidence
    return sentence_df


def summarize_sections(doc: DocumentData, window_df: pd.DataFrame, sentence_df: pd.DataFrame, config: PipelineConfig) -> List[Dict[str, Any]]:
    summaries = []
    window_group = {sid: grp for sid, grp in (window_df.groupby("section_id") if not window_df.empty else [])}
    sentence_group = {sid: grp for sid, grp in (sentence_df.groupby("section_id") if not sentence_df.empty else [])}
    for section in doc.sections:
        wdf = window_group.get(section.section_id, pd.DataFrame())
        sdf = sentence_group.get(section.section_id, pd.DataFrame())
        coverage_ai = float(((wdf["window_label"] == "ai") & (wdf["window_confidence"] >= config.confidence_threshold)).mean()) if not wdf.empty else 0.0
        coverage_para = float(((wdf["window_label"] == "paraphrased_ai") & (wdf["window_confidence"] >= config.confidence_threshold)).mean()) if not wdf.empty else 0.0
        mean_probs = {label: float(wdf[label].mean()) if not wdf.empty else 0.0 for label in PROB_COLUMNS}
        top_window = None
        if not wdf.empty:
            top_idx = wdf["window_confidence"].idxmax()
            top_row = wdf.loc[top_idx]
            top_window = {
                "label": top_row["window_label"],
                "confidence": float(top_row["window_confidence"]),
                "text": top_row["text_raw"],
                "span": [int(top_row["span_start"]), int(top_row["span_end"])],
            }
        summaries.append({
            "section_id": section.section_id,
            "title": section.title,
            "normalized": section.normalized,
            "num_windows": int(len(wdf)),
            "num_sentences": int(len(sdf)),
            "coverage_ai": coverage_ai,
            "coverage_paraphrased": coverage_para,
            "probabilities": mean_probs,
            "top_window": top_window,
            "flags": section.flags,
        })
    return summaries


def determine_verdict(doc_probs: Dict[str, float], coverage_ai: float, coverage_para: float, max_sim: float, detectgpt_mean: float, config: PipelineConfig) -> Tuple[str, str]:
    pa = doc_probs.get("paraphrased_ai", 0.0)
    ai = doc_probs.get("ai", 0.0)
    if pa >= 0.5 or (coverage_para >= config.coverage_para_threshold and max_sim >= config.tau_semantic):
        return "paraphrased_ai", "Paraphrased AI (Likely)"
    if ai >= 0.5 or (coverage_ai >= config.coverage_ai_threshold and detectgpt_mean <= config.detectgpt_threshold):
        return "ai", "AI-generated (Likely)"
    return "human", "Human (Likely)"


def build_highlight_html(sentence_df: pd.DataFrame) -> str:
    if sentence_df.empty:
        return "<p>No narrative text detected.</p>"
    parts = ["<div style='line-height:1.6;font-size:16px'>"]
    current_section = None
    for row in sentence_df.itertuples():
        if current_section != row.section_id:
            if current_section is not None:
                parts.append("<br/>")
            heading = row.section_name or f"Section {row.section_id + 1}"
            parts.append(f"<div style='font-weight:bold;margin-top:1em'>{heading}</div>")
            current_section = row.section_id
        color = COLOR_MAP.get(row.label, "#6b7280")
        title = f"{row.label} ({row.confidence:.2f})"
        safe_text = row.text.replace("<", "&lt;").replace(">", "&gt;")
        if row.block_type == getattr(processing, "BLOCK_LIST", "LIST"):
            parts.append(f"<div style='margin-left:1em'>&#8226; <span style='background-color:{color}22;border-bottom:2px solid {color};padding:1px 2px' title='{title}'>{safe_text}</span></div>")
        else:
            parts.append(f"<span style='background-color:{color}22;border-bottom:2px solid {color};padding:1px 2px;margin-right:4px' title='{title}'>{safe_text}</span>")
    parts.append("</div>")
    return "".join(parts)


def build_export_payload(result: AnalysisResult) -> Dict[str, Any]:
    doc = result.document_summary
    payload = {
        "document": doc,
        "sections": result.section_summaries,
        "windows": result.window_df.to_dict(orient="records"),
        "sentences": result.sentence_df.to_dict(orient="records"),
    }
    return payload


def compute_meta_probs(window_df: pd.DataFrame, meta_bundle, config: PipelineConfig) -> Optional[Dict[str, float]]:
    if not meta_bundle or window_df.empty:
        return None
    model, feat_cols, mu, sigma, label_map = meta_bundle
    if model is None or not feat_cols:
        return None
    values = []
    for name in feat_cols:
        if name == "num_windows":
            values.append(float(len(window_df)))
            continue
        if name == "coverage":
            if "max_sim" in window_df:
                values.append(float((window_df["max_sim"] >= config.tau_semantic).mean()))
            else:
                values.append(0.0)
            continue
        if name.endswith("_mean"):
            base = name[:-5]
            if base in window_df:
                values.append(float(window_df[base].mean()))
                continue
        if name.endswith("_max"):
            base = name[:-4]
            if base in window_df:
                values.append(float(window_df[base].max()))
                continue
        if name in window_df:
            values.append(float(window_df[name].mean()))
        else:
            values.append(0.0)
    vec = np.array(values, dtype=np.float32)
    if mu is not None and sigma is not None and len(mu) == len(vec):
        sigma_safe = sigma.copy()
        sigma_safe[sigma_safe == 0] = 1.0
        vec = (vec - mu) / sigma_safe
    try:
        with torch.no_grad():
            logits = model(torch.from_numpy(vec).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    except Exception:
        return None
    order = [label_map[k] for k in sorted(label_map.keys())]
    out = {normalize_label_name(name): float(probs[idx]) for idx, name in enumerate(order)}
    for lbl in DEFAULT_LABEL_ORDER:
        out.setdefault(lbl, 0.0)
    total = sum(out.values())
    if total > 0:
        for key in out:
            out[key] /= total
    return out


# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------
def run_pipeline(doc: DocumentData, config: PipelineConfig, resources: Dict[str, Any]) -> AnalysisResult:
    alerts: List[str] = []
    if not doc.windows:
        sentence_df = build_sentence_df(doc)
        if not sentence_df.empty:
            for col in PROB_COLUMNS:
                sentence_df[col] = 1.0 if col == "p_human" else 0.0
            sentence_df["label"] = "human"
            sentence_df["confidence"] = 0.0
        highlight = build_highlight_html(sentence_df)
        doc_summary = {
            "verdict_label": "human",
            "verdict_text": "Human (Likely)",
            "probabilities": {lbl: (1.0 if lbl == "human" else 0.0) for lbl in DEFAULT_LABEL_ORDER},
            "coverage_ai": 0.0,
            "coverage_paraphrased": 0.0,
            "max_sim": 0.0,
            "detectgpt_mean": 0.0,
            "percentages": {lbl: (100.0 if lbl == "human" else 0.0) for lbl in DEFAULT_LABEL_ORDER},
        }
        payload = build_export_payload(AnalysisResult(doc, pd.DataFrame(), sentence_df, [], doc_summary, alerts, highlight, {}))
        return AnalysisResult(doc, pd.DataFrame(), sentence_df, [], doc_summary, alerts, highlight, payload)

    window_ids = [w.window_id for w in doc.windows]
    window_texts = [w.text_clean for w in doc.windows]
    base_rows = []
    for w in doc.windows:
        base_rows.append({
            "window_id": w.window_id,
            "section_id": w.section_id,
            "section_name": w.section_name,
            "span_start": w.span_start,
            "span_end": w.span_end,
            "sentence_indices": w.sentence_indices,
            "text_raw": w.text_raw,
            "text_clean": w.text_clean,
        })
    window_df = pd.DataFrame(base_rows)

    deberta_df = compute_deberta_probs(window_texts, window_ids, alerts)
    styl_df = compute_stylometry(window_texts, window_ids, alerts)
    ppl_df = compute_perplexity(window_texts, window_ids, resources.get("perplexity"), alerts)
    sem_df, sem_doc = compute_semantic(window_texts, window_ids, resources.get("semantic", (None, None)), config.tau_semantic, alerts)

    frames = [window_df, deberta_df]
    for df in (styl_df, ppl_df, sem_df):
        if df is not None and not df.empty:
            frames.append(df)
    window_df = frames[0]
    for extra in frames[1:]:
        window_df = window_df.merge(extra, on="window_id", how="left")

    meta_window_probs = compute_meta_window_probs(window_df, meta_bundle, sem_doc) if meta_bundle else None
    for col in PROB_COLUMNS:
        if col not in window_df:
            window_df[col] = 0.0
    if meta_window_probs is not None:
        for col in PROB_COLUMNS:
            window_df[col] = meta_window_probs[col]
    raw_probs = window_df[PROB_COLUMNS].to_numpy(dtype=np.float32)
    smoothed = smooth_probabilities(raw_probs)
    for idx, col in enumerate(PROB_COLUMNS):
        window_df[f"{col}_raw"] = window_df[col]
        window_df[col] = smoothed[:, idx]
    window_df["window_label"] = [DEFAULT_LABEL_ORDER[row.argmax()] for row in smoothed]
    window_df["window_confidence"] = smoothed.max(axis=1)

    sentence_df = compute_sentence_probabilities(build_sentence_df(doc), window_df)
    section_summaries = summarize_sections(doc, window_df, sentence_df, config)

    coverage_ai = float(((window_df["window_label"] == "ai") & (window_df["window_confidence"] >= config.confidence_threshold)).mean())
    coverage_para = float(((window_df["window_label"] == "paraphrased_ai") & (window_df["window_confidence"] >= config.confidence_threshold)).mean())
    max_sim = float(window_df["max_sim"].max()) if "max_sim" in window_df else 0.0
    detectgpt_mean = float(window_df["detectgpt_score"].mean()) if "detectgpt_score" in window_df else 0.0
    if meta_window_probs is not None:
        avg_probs = {lbl: float(meta_window_probs[f"p_{lbl}"].mean()) for lbl in DEFAULT_LABEL_ORDER}
    else:
        avg_probs = {lbl: float(window_df[f"p_{lbl}"].mean()) for lbl in DEFAULT_LABEL_ORDER}

    meta_probs = compute_meta_probs(window_df, resources.get("meta"), config)
    doc_probs = meta_probs or avg_probs
    verdict_label, verdict_text = determine_verdict(doc_probs, coverage_ai, coverage_para, max_sim, detectgpt_mean, config)

    sentence_percentages = {}
    if not sentence_df.empty:
        for lbl in DEFAULT_LABEL_ORDER:
            sentence_percentages[lbl] = float((sentence_df["label"] == lbl).mean() * 100.0)
    else:
        sentence_percentages = {lbl: (100.0 if lbl == "human" else 0.0) for lbl in DEFAULT_LABEL_ORDER}

    document_summary = {
        "verdict_label": verdict_label,
        "verdict_text": verdict_text,
        "probabilities": doc_probs,
        "coverage_ai": coverage_ai,
        "coverage_paraphrased": coverage_para,
        "max_sim": max_sim,
        "detectgpt_mean": detectgpt_mean,
        "percentages": sentence_percentages,
        "semantic_doc": sem_doc,
    }

    highlight_html = build_highlight_html(sentence_df)
    result = AnalysisResult(doc, window_df, sentence_df, section_summaries, document_summary, alerts, highlight_html, {})
    result.export_payload = build_export_payload(result)
    return result


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Text Detector (Scientific Papers)", layout="wide")
st.title("AI Text Detector - Turnitin-style (Scientific Papers)")

processor = get_processor()
perplexity_model = get_perplexity_analyzer()
semantic_resources = load_semantic_resources(AI_INDEX_DIR, EMB_MODEL)
meta_bundle = load_meta_resources(META_CKPT, META_COLS)

with st.sidebar:
    st.header("Run Settings")
    tau_setting = st.slider("Paraphrase t", 0.6, 0.95, DEFAULT_TAU, 0.01)
    max_pages = st.number_input("Max PDF pages", min_value=1, max_value=200, value=DEFAULT_MAX_PAGES, step=1)
    st.caption("Tune t on validation data. Tables and figures are skipped for stylometry/perplexity.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
text_input = st.text_area("Or paste scientific text", height=220)
run_button = st.button("Analyze")

analysis_state = st.session_state.get("last_analysis")

if run_button:
    doc_bytes = None
    if uploaded is None and not text_input.strip():
        st.warning("Please upload a PDF or paste text to analyze.")
        st.stop()
    try:
        with st.spinner("Ingesting document..."):
            if uploaded is not None:
                doc_bytes = uploaded.read()
                doc = processor.from_pdf(doc_bytes, doc_id=uploaded.name or "uploaded_pdf", max_pages=int(max_pages))
            else:
                doc = processor.from_text(text_input, doc_id="pasted_text")
    except Exception as exc:
        st.error(f"Failed to ingest document: {exc}")
        st.stop()
    config = PipelineConfig(tau_semantic=tau_setting)
    resources = {
        "perplexity": perplexity_model,
        "semantic": semantic_resources,
        "meta": meta_bundle,
    }
    with st.spinner("Running detectors..."):
        result = run_pipeline(doc, config, resources)
    analysis_state = {
        "doc": doc,
        "result": result,
        "config": config,
        "pdf_bytes": doc_bytes if uploaded is not None else None,
        "source_name": uploaded.name if uploaded is not None else "pasted_text",
        "highlight_pdf": None,
    }
    st.session_state["last_analysis"] = analysis_state

if analysis_state:
    doc = analysis_state["doc"]
    result = analysis_state["result"]
    config = analysis_state["config"]
    for msg in result.alerts:
        st.warning(msg)
    summary = result.document_summary
    verdict_label = summary.get("verdict_label", "human")
    verdict_text = summary.get("verdict_text", "Human (Likely)")
    if verdict_label == "human":
        st.success(verdict_text)
    elif verdict_label == "ai":
        st.error(verdict_text)
    else:
        st.warning(verdict_text)
    prob_cols = st.columns(3)
    prob_values = summary.get("probabilities", {})
    for col, label in zip(prob_cols, DEFAULT_LABEL_ORDER):
        col.metric(f"P({label})", f"{prob_values.get(label, 0.0):.2f}")
    pct_cols = st.columns(3)
    percent_values = summary.get("percentages", {})
    for col, label in zip(pct_cols, DEFAULT_LABEL_ORDER):
        col.metric(f"{label.title()} %", f"{percent_values.get(label, 0.0):.1f}%")

    st.markdown("### Class Distribution")
    pie_html = build_pie_chart_html(percent_values)
    st.markdown(pie_html, unsafe_allow_html=True)

    st.markdown("### Highlighted Document")
    st.markdown(build_color_legend_html(), unsafe_allow_html=True)
    st.markdown(result.highlight_html, unsafe_allow_html=True)

    pdf_source = analysis_state.get("pdf_bytes")
    if pdf_source and fitz is not None:
        if analysis_state.get("highlight_pdf") is None:
            analysis_state["highlight_pdf"] = generate_highlighted_pdf(pdf_source, result.sentence_df)
            st.session_state["last_analysis"] = analysis_state
        highlight_pdf = analysis_state.get("highlight_pdf")
        if highlight_pdf:
            st.download_button(
                "Download highlighted PDF",
                data=highlight_pdf,
                file_name=f"{analysis_state.get('source_name', 'document')}_highlighted.pdf",
                mime="application/pdf",
            )

    pie_html = build_pie_chart_html(percent_values)
    legend_html = build_color_legend_html()
    download_html = f"<html><body>{legend_html}{pie_html}{result.highlight_html}</body></html>"
    st.download_button(
        "Download highlighted HTML",
        data=download_html.encode("utf-8"),
        file_name="ai_detection_report.html",
        mime="text/html",
    )

    st.download_button(
        "Download JSON (spans + decisions)",
        data=json.dumps(result.export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="ai_detection_report.json",
        mime="application/json",
    )

    st.markdown("### Section Summary")
    section_table = pd.DataFrame(result.section_summaries)
    if not section_table.empty:
        st.dataframe(section_table[["section_id", "title", "normalized", "num_windows", "coverage_ai", "coverage_paraphrased"]])
    else:
        st.info("No sections with narrative text were detected.")

    with st.expander("Window-level diagnostics"):
        st.dataframe(result.window_df)

    with st.expander("Sentence-level scores"):
        st.dataframe(result.sentence_df[["sentence_id", "section_name", "label", "confidence", "text"]])
else:
    st.info("Upload a PDF or paste text to begin analysis.")
