# --- required imports (put at top of your file) ---
import re, random, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from nltk.corpus import wordnet as wn
    _HAS_WN = True
except Exception:
    _HAS_WN = False

# expect DEVICE already defined globally; else:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerplexityEntropyAnalyzer:
    """
    Computes:
      - Perplexity (PPL) = exp(mean NLL)
      - Entropy (mean next-token entropy across positions)
      - Surprisal stats (-log p(true_token | context)): mean/var
      - Avg rank of true tokens
      - DetectGPT-style score: logp(text) - mean(logp(perturbed_texts))
    Notes:
      * DetectGPT here uses cheap lexical/noise perturbations; swap in a paraphraser if desired.
      * For speed, perturbations are batched across all inputs.
    """

    def __init__(self,
                 model_name: str = "gpt2-medium",
                 max_len: int = 512,
                 batch_size: int = 8,
                 top_k_entropy: int = None):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.top_k_entropy = top_k_entropy
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    # ----------------- Core scoring -----------------
    @torch.no_grad()
    def _score_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        enc = self.tok(texts, truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt").to(DEVICE)

        out = self.model(**enc)
        logits = out.logits[:, :-1, :]        # (B, T-1, V)
        labels = enc["input_ids"][:, 1:]      # (B, T-1)
        attn   = enc["attention_mask"][:, 1:] # (B, T-1)

        logprobs = F.log_softmax(logits, dim=-1)              # (B, T-1, V)
        true_lp  = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        mask     = attn.bool()
        tok_counts = mask.sum(dim=1).clamp_min(1)

        # Entropy: full or top-k approx
        if self.top_k_entropy is None:
            probs = logprobs.exp()
            entropy = -(probs * logprobs).sum(dim=-1)         # (B, T-1)
        else:
            topk_vals, _ = torch.topk(logits, k=self.top_k_entropy, dim=-1)
            topk_logprobs = F.log_softmax(topk_vals, dim=-1)
            topk_probs    = topk_logprobs.exp()
            entropy = -(topk_probs * topk_logprobs).sum(dim=-1)

        # Rank of true token
        true_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        greater = (logits > true_logits.unsqueeze(-1)).sum(dim=-1)
        rank = greater + 1

        # Aggregates
        nll = -(true_lp * mask).sum(dim=1) / tok_counts
        ppl = torch.exp(nll)
        mean_entropy = (entropy * mask).sum(dim=1) / tok_counts
        surprisal = -(true_lp)
        surprisal_mean = (surprisal * mask).sum(dim=1) / tok_counts
        s2 = ((surprisal - surprisal_mean.unsqueeze(1))**2 * mask).sum(dim=1) / tok_counts
        avg_rank = (rank.float() * mask).sum(dim=1) / tok_counts

        return {
            "ppl": ppl.detach().cpu().numpy(),
            "mean_entropy": mean_entropy.detach().cpu().numpy(),
            "surprisal_mean": surprisal_mean.detach().cpu().numpy(),
            "surprisal_var": s2.detach().cpu().numpy(),
            "avg_rank": avg_rank.detach().cpu().numpy(),
            "logprob": (-nll).detach().cpu().numpy(),  # mean log-prob per token
        }

    @torch.no_grad()
    def score_texts(self, texts: List[str], show_progress=True) -> pd.DataFrame:
        rows = []
        idxs = range(0, len(texts), self.batch_size)
        if show_progress:
            idxs = tqdm(idxs, desc="Scoring PPL/entropy")
        for i in idxs:
            batch = texts[i:i+self.batch_size]
            s = self._score_batch(batch)
            for j in range(len(batch)):
                rows.append({k: s[k][j] for k in s.keys()})
        return pd.DataFrame(rows)

    # ----------------- DetectGPT-lite -----------------
    def _wordnet_synonyms(self, word: str) -> List[str]:
        if not _HAS_WN:
            return []
        outs = set()
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                w = l.name().replace('_', ' ')
                if w.lower() != word.lower() and w.isascii() and len(w) > 1:
                    outs.add(w)
        # cap to prevent exploding length
        return list(outs)[:3]

    def _perturb_once(self, text: str,
                      p_drop=0.05, p_swap=0.05, p_syn=0.05, rng=None) -> str:
        if rng is None: rng = random
        toks = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        n = len(toks)
        if n == 0: return text

        # synonym replace
        if _HAS_WN and p_syn > 0:
            for i in range(n):
                if toks[i].isalpha() and rng.random() < p_syn:
                    syns = self._wordnet_synonyms(toks[i])
                    if syns: toks[i] = rng.choice(syns)

        # drop tokens
        kept = []
        for t in toks:
            if (t.isalpha() or t.isdigit()) and rng.random() < p_drop:
                continue
            kept.append(t)
        toks = kept

        # swap adjacents
        i = 0
        while i < len(toks)-1:
            if (toks[i].isalnum() and toks[i+1].isalnum()) and rng.random() < p_swap:
                toks[i], toks[i+1] = toks[i+1], toks[i]
                i += 2
            else:
                i += 1
        return " ".join(toks)

    @torch.no_grad()
    def detectgpt_score(self, texts: List[str], K: int = 8,
                        p_drop=0.05, p_swap=0.05, p_syn=0.05, seed=42,
                        show_progress=True) -> np.ndarray:
        """
        score = logp(x) - mean_k logp(perturb_k(x))
        More negative => more AI-like per DetectGPT intuition.
        Batched across all perturbations for speed.
        """
        # base scores
        base_df = self.score_texts(texts, show_progress=show_progress)
        base_lp = base_df["logprob"].values

        # build all perturbations first (batched scoring)
        rng = random.Random(seed)
        all_pts = []
        owners  = []  # map perturbation -> original idx
        for idx, t in enumerate(texts):
            for _ in range(K):
                all_pts.append(self._perturb_once(t, p_drop=p_drop, p_swap=p_swap, p_syn=p_syn, rng=rng))
                owners.append(idx)

        # score all perturbations in batches
        neigh_lp = np.zeros((len(texts), K), dtype=np.float32)
        idxs = range(0, len(all_pts), self.batch_size)
        if show_progress:
            idxs = tqdm(idxs, desc=f"DetectGPT perturb ({K}×)")
        cursor = 0
        for i in idxs:
            batch = all_pts[i:i+self.batch_size]
            s = self._score_batch(batch)
            B = len(batch)
            for j in range(B):
                orig_idx = owners[cursor + j]
                slot = (cursor + j) % K
                neigh_lp[orig_idx, slot] = s["logprob"][j]
            cursor += B

        mean_neighbor_lp = neigh_lp.mean(axis=1)
        return base_lp - mean_neighbor_lp

    # ----------------- One-call convenience -----------------
    def features(self, texts: List[str], K_detectgpt=8, show_progress=True) -> pd.DataFrame:
        base = self.score_texts(texts, show_progress=show_progress)
        dg = self.detectgpt_score(texts, K=K_detectgpt, show_progress=show_progress)
        base["detectgpt_score"] = dg
        return base
