import os, re, json, math, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
import faiss
from transformers import AutoTokenizer, AutoModel
from typing import Optional, List, Dict, Any, Tuple


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Sentence windows (3-sentence default) --------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str):
    if not text: return []
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]

def windows_from_text(text: str, win=3, stride=2):
    sents = split_sentences(text)
    out = []
    i = 0
    while i < len(sents):
        j = min(len(sents), i+win)
        out.append({
            "span_start": i,
            "span_end": j-1,
            "text": " ".join(sents[i:j])
        })
        if j == len(sents): break
        i += stride
    return out

# -------- E5 embedder (HF only, mean-pooling) --------
class E5Embedder:
    """
    Default model is small/fast. If you have more VRAM, switch to 'intfloat/e5-base-v2' or 'bge-base-en-v1.5'.
    NOTE: E5 expects 'query: ' and 'passage: ' prefixes; we use 'passage:' for corpus and 'query:' for queries.
    """
    def __init__(self, model_name="intfloat/e5-small-v2", max_len=256, batch_size=64):
        self.model_name = model_name
        self.max_len = max_len
        self.bs = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def _encode(self, texts, is_query=False):
        prefix = "query: " if is_query else "passage: "
        # E5 wants lowercase prefixes exactly like this
        texts = [prefix + (t or "") for t in texts]
        enc = self.tok(
            texts, padding=True, truncation=True, max_length=self.max_len,
            return_tensors="pt"
        ).to(DEVICE)
        out = self.model(**enc)
        # mean pooling with attention mask
        last = out.last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        # L2-normalize for cosine/IP search
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().numpy()

    def encode(self, texts, is_query=False):
        vecs = []
        for i in tqdm(range(0, len(texts), self.bs), desc=f"Embed[{self.model_name}]"):
            vecs.append(self._encode(texts[i:i+self.bs], is_query=is_query))
        return np.vstack(vecs) if vecs else np.empty((0, self.model.config.hidden_size), dtype=np.float32)

# -------- FAISS index wrapper (cosine via inner product on L2-normalized vectors) --------
class SemanticIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # IP on normalized vectors == cosine
        self.meta = []                      # parallel list of dicts
        self.ntotal = 0

    def add(self, vectors: np.ndarray, metadata: list):
        assert vectors.dtype == np.float32
        self.index.add(vectors)
        self.meta.extend(metadata)
        self.ntotal += vectors.shape[0]

    def search(self, qvecs: np.ndarray, topk=5):
        sims, idxs = self.index.search(qvecs.astype(np.float32), topk)
        # Return (sims, metas) where metas is list of list of dicts
        metas = [[self.meta[i] if i != -1 else None for i in row] for row in idxs]
        return sims, metas

    def save(self, path_dir: str):
        os.makedirs(path_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path_dir, "index.faiss"))
        with open(os.path.join(path_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path_dir: str):
        idx = faiss.read_index(os.path.join(path_dir, "index.faiss"))
        meta=[]
        with open(os.path.join(path_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
            for line in f: meta.append(json.loads(line))
        obj = cls(idx.d)
        obj.index = idx
        obj.meta = meta
        obj.ntotal = idx.ntotal
        return obj

# -------- Query a document and compute paraphrase features --------
def search_windows_for_paraphrase(
    window_texts,
    index_dir: str,
    embedder_model="intfloat/e5-small-v2",
    max_len=256,
    batch_size=64,
    topk=5,
    tau=0.78,
    embedder: Optional[E5Embedder] = None,
    index: Optional[SemanticIndex] = None,
):
    """
    Compute paraphrase similarity features for a list of already prepared window texts.
    Returns a dataframe with per-window scores and aggregated document-level metrics.
    """
    if not window_texts:
        empty = pd.DataFrame(columns=["window_index", "max_sim", "mean_top3", "above_tau", "nn_meta"])
        doc_features = {"coverage": 0.0, "max_sim": 0.0, "top3_mean": 0.0, "frac_above_tau": 0.0, "num_windows": 0}
        return empty, doc_features
    embedder = embedder or E5Embedder(embedder_model, max_len=max_len, batch_size=batch_size)
    index = index or SemanticIndex.load(index_dir)
    qvecs = embedder.encode(list(window_texts), is_query=True).astype(np.float32)
    sims, metas = index.search(qvecs, topk=topk)
    rows = []
    above = 0
    top3_means = []
    max_vals = []
    for idx, sim_row in enumerate(sims):
        if len(sim_row) == 0:
            max_sim = 0.0
            mean_top3 = 0.0
        else:
            max_sim = float(sim_row[0])
            mean_top3 = float(np.mean(sim_row[: min(3, len(sim_row))]))
        flag = 1 if max_sim >= tau else 0
        above += flag
        top3_means.append(mean_top3)
        max_vals.append(max_sim)
        rows.append({
            "window_index": idx,
            "max_sim": max_sim,
            "mean_top3": mean_top3,
            "above_tau": flag,
            "nn_meta": metas[idx][0] if metas[idx] else None,
        })
    df = pd.DataFrame(rows)
    doc_features = {
        "coverage": float(above / len(window_texts)),
        "max_sim": float(max(max_vals) if max_vals else 0.0),
        "top3_mean": float(np.mean(top3_means) if top3_means else 0.0),
        "frac_above_tau": float(above / len(window_texts)),
        "num_windows": len(window_texts),
    }
    return df, doc_features

def search_document_for_paraphrase(doc_text: str,
                                   index_dir: str,
                                   embedder_model="intfloat/e5-small-v2",
                                   max_len=256, batch_size=64,
                                   win=3, stride=2,
                                   topk=5, tau=0.78,
                                   embedder: Optional[E5Embedder] = None,
                                   index: Optional[SemanticIndex] = None):
    """
    Convenience wrapper that segments doc_text into windows and delegates to
    search_windows_for_paraphrase so callers that do not have pre-built windows
    can reuse the same FAISS features.
    """
    wins = windows_from_text(doc_text, win=win, stride=stride)
    if not wins:
        empty_df = pd.DataFrame([])
        doc_features = {
            "coverage": 0.0,
            "max_sim": 0.0,
            "top3_mean": 0.0,
            "frac_above_tau": 0.0,
            "num_windows": 0,
        }
        return empty_df, doc_features
    window_texts = [w["text"] for w in wins]
    window_df, doc_features = search_windows_for_paraphrase(
        window_texts,
        index_dir=index_dir,
        embedder_model=embedder_model,
        max_len=max_len,
        batch_size=batch_size,
        topk=topk,
        tau=tau,
        embedder=embedder,
        index=index,
    )
    if window_df.empty:
        return window_df, doc_features
    span_start = []
    span_end = []
    texts = []
    for idx in window_df["window_index"].astype(int).tolist():
        win = wins[idx]
        span_start.append(win["span_start"])
        span_end.append(win["span_end"])
        texts.append(win["text"])
    window_df = window_df.assign(span_start=span_start, span_end=span_end, text=texts)
    return window_df, doc_features
