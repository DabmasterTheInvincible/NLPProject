import re, math, statistics, numpy as np, pandas as pd
from typing import List, Optional
from tqdm.auto import tqdm
# -------- Speed/behavior knobs --------
WINDOW_SIZE = 10**9   # big enough to swallow any doc
STRIDE = 10**9                # move window by 2 sentences (use 1 for max detail, slower)
USE_SPACY = False        # set True only if you really need better POS (slower)
DO_READABILITY = True    # set False to skip readability for speed
MATTR_WINDOW = 50        # MATTR window (tokens); reduce to 25 for speed
MAX_DOCS = None          # e.g., 5000 to cap docs for a quick run; or None
TEXT_COL = "text"
ID_COL = "id"            # if not present, numeric index will be used

# -------- Fast tokenization & sentence split --------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_WORD_RE    = re.compile(r"[A-Za-z][A-Za-z'\-]*|[0-9]+(?:\.[0-9]+)?")
_VOWELS     = re.compile(r"[aeiouyAEIOUY]+")

def split_sentences_fast(text: str):
    if not text: return []
    sents = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sents if s.strip()]

def tokenize_fast(text: str):
    return _WORD_RE.findall(text or "")

# -------- Readability (cheap approximations) --------
def approx_syllables(text: str):
    # Approximate by vowel-group count minus common silent-e effects (good enough for relative scores)
    if not text: return 0
    # remove obvious email/urls/numbers to avoid huge counts
    t = re.sub(r"(https?://\S+)|(\w+@\w+\.\w+)|\d[\d\.\-]*", " ", text)
    syl = len(_VOWELS.findall(t))
    return max(1, syl)

def flesch(sents, words, syllables):
    if not sents or not words: return 0.0
    return 206.835 - 1.015*(words/sents) - 84.6*(syllables/words)

def fk_grade(sents, words, syllables):
    if not sents or not words: return 0.0
    return 0.39*(words/sents) + 11.8*(syllables/words) - 15.59

def gunning_fog(sents, words, complex_words):
    if not sents or not words: return 0.0
    return 0.4 * ((words/sents) + 100.0 * (complex_words / words))

def count_complex(words):
    # "complex" ~ >=3 syllables via quick heuristic: 3+ vowel groups
    return sum(1 for w in words if len(_VOWELS.findall(w)) >= 3)

# -------- Lexical diversity --------
def ttr(tokens):
    n = len(tokens)
    return (len(set(tokens)) / n) if n else 0.0

def mattr(tokens, window=MATTR_WINDOW):
    n = len(tokens)
    if n == 0: return 0.0
    if n <= window: return ttr(tokens)
    scores = []
    for i in range(0, n - window + 1):
        scores.append(ttr(tokens[i:i+window]))
    return float(np.mean(scores)) if scores else ttr(tokens)

# -------- POS ratios (heuristic: fast and dependency-free) --------
PRON = set("i you he she it we they me him her us them my your his their our mine yours hers theirs ours".split())
DET  = set("a an the this that these those each every some any".split())
CCONJ= set("and or nor but yet so".split())
ADP  = set("of in to for with on at from by about as into like through after over between out against during without before under around among".split())

def guess_pos(word):
    w = word.lower()
    if w in PRON:  return "PRON"
    if w in DET:   return "DET"
    if w in CCONJ: return "CCONJ"
    if w in ADP:   return "ADP"
    if w.endswith("ly"):        return "ADV"
    if w.endswith(("ed","ing")):return "VERB"
    if w.endswith(("ous","ful","able","al","ive","ic")): return "ADJ"
    if re.search(r"(ness|tion|ment|ity|ship|ism|age|ery|ance|ence)$", w): return "NOUN"
    return "NOUN"

def pos_ratios_heuristic(tokens):
    wanted = ["NOUN","VERB","ADJ","ADV","PRON","DET","ADP","AUX","CCONJ"]
    cnt = {k:0 for k in wanted + ["OTHER"]}
    for t in tokens:
        tag = guess_pos(t)
        cnt[tag] = cnt.get(tag, 0) + 1
    total = sum(cnt.values()) or 1
    return {f"pos_{k.lower()}": cnt[k]/total for k in cnt}

# (Optional) spaCy path — disabled by default (slow)
if USE_SPACY:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        HAVE_SPACY = True
    except Exception:
        nlp = spacy.blank("en")
        HAVE_SPACY = False
else:
    nlp = None
    HAVE_SPACY = False

def pos_ratios(text, tokens):
    if HAVE_SPACY:
        doc = nlp(text)
        wanted = ["NOUN","VERB","ADJ","ADV","PRON","DET","ADP","AUX","CCONJ"]
        cnt = {k:0 for k in wanted + ["OTHER"]}
        for t in doc:
            if not t.text.strip(): continue
            tag = t.pos_ if t.pos_ else "OTHER"
            if tag in cnt: cnt[tag]+=1
            else: cnt["OTHER"]+=1
        total = sum(cnt.values()) or 1
        return {f"pos_{k.lower()}": cnt[k]/total for k in cnt}
    else:
        return pos_ratios_heuristic(tokens)

# -------- Burstiness over sentence lengths --------
def burstiness(sentences):
    lens = [len(tokenize_fast(s)) for s in sentences if s]
    if len(lens) <= 1:
        return {"burst_b": 0.0, "burst_cv": 0.0, "mean_slen": float(lens[0] if lens else 0.0)}
    mu = statistics.mean(lens)
    sigma = statistics.pstdev(lens)
    b = (sigma - mu) / (sigma + mu + 1e-12)
    cv = sigma / (mu + 1e-12)
    return {"burst_b": float(b), "burst_cv": float(cv), "mean_slen": float(mu)}

# -------- Core window feature extraction --------
def features_for_text(text, doc_id, window_size=WINDOW_SIZE, stride=STRIDE):
    sents = split_sentences_fast(text)
    n = len(sents)
    if n == 0:
        return []
    rows = []
    start = 0
    while start < n:
        end = min(n, start + window_size)
        win_sents = sents[start:end]
        sub = " ".join(win_sents)
        toks = [t.lower() for t in tokenize_fast(sub)]
        wcnt = len(toks); scnt = len(win_sents)

        # Lexical
        row = {
            "id": doc_id,
            "span_start": start,
            "span_end":   end-1,
            "n_sent": scnt,
            "n_words": wcnt,
            "avg_word_len": (sum(len(t) for t in toks)/wcnt) if wcnt else 0.0,
            "ttr": ttr(toks),
            "mattr": mattr(toks, window=MATTR_WINDOW),
        }

        # POS
        row.update(pos_ratios(sub, toks))

        # Burstiness in this window (based on the window's sentences)
        row.update(burstiness(win_sents))

        # Readability (approximate, fast)
        if DO_READABILITY:
            syll = approx_syllables(sub)
            cmplx = count_complex(toks)
            row["flesch"]    = flesch(scnt, wcnt, syll)
            row["fk_grade"]  = fk_grade(scnt, wcnt, syll)
            row["gunning_fog"]= gunning_fog(scnt, wcnt, cmplx)

        rows.append(row)
        if end == n: break
        start += stride
    return rows

def run_stylometry(df, text_col=TEXT_COL, id_col=ID_COL, max_docs=MAX_DOCS):
    feats = []
    it = df.itertuples(index=False)
    total = len(df) if max_docs is None else min(max_docs, len(df))
    for i, row in tqdm(enumerate(it), total=total, desc="Stylometry"):
        if max_docs is not None and i >= max_docs: break
        text = getattr(row, text_col)
        doc_id = getattr(row, id_col) if (id_col and hasattr(row, id_col)) else i
        feats.extend(features_for_text(text, doc_id))
    return pd.DataFrame(feats)
def features_for_windows(texts: List[str], window_ids: Optional[List[int]] = None) -> pd.DataFrame:
    rows = []
    for idx, text in enumerate(texts):
        tokens = [t.lower() for t in tokenize_fast(text)]
        sents = split_sentences_fast(text)
        n_words = len(tokens)
        n_sent = len(sents)
        row = {
            "window_id": window_ids[idx] if window_ids is not None else idx,
            "n_sent": n_sent,
            "n_words": n_words,
            "avg_word_len": (sum(len(t) for t in tokens) / n_words) if n_words else 0.0,
            "ttr": ttr(tokens),
            "mattr": mattr(tokens, window=MATTR_WINDOW),
        }
        row.update(pos_ratios(text, tokens))
        row.update(burstiness(sents if sents else [text]))
        if DO_READABILITY:
            syll = approx_syllables(text)
            cmplx = count_complex(tokens)
            row["flesch"] = flesch(n_sent, n_words, syll)
            row["fk_grade"] = fk_grade(n_sent, n_words, syll)
            row["gunning_fog"] = gunning_fog(n_sent, n_words, cmplx)
        rows.append(row)
    return pd.DataFrame(rows)
