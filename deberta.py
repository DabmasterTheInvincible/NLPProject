import os
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_DIR = os.environ.get("DEBERTA_DIR", "./models/deberta")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LABELS = {0: "human", 1: "ai", 2: "paraphrased_ai"}

_tokenizer = None
_model = None
_id2label = None


def load(model_dir: str = DEFAULT_DIR):
    global _tokenizer, _model, _id2label
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        _model.eval()
        config = _model.config
        if getattr(config, "id2label", None):
            _id2label = {int(k): v for k, v in config.id2label.items()}
        else:
            _id2label = DEFAULT_LABELS.copy()
    return _tokenizer, _model, _id2label


def predict(texts: List[str], max_length: int = 320, batch_size: int = 8, model_dir: str = DEFAULT_DIR) -> Tuple[List[str], np.ndarray]:
    if not texts:
        num_labels = len(_id2label) if _id2label is not None else len(DEFAULT_LABELS)
        return [], np.zeros((0, num_labels), dtype=np.float32)
    tokenizer, model, id2label = load(model_dir)
    all_probs: List[np.ndarray] = []
    label_ids: List[int] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        label_ids.extend(probs.argmax(axis=1).tolist())
    stacked = np.vstack(all_probs) if all_probs else np.zeros((0, len(id2label)), dtype=np.float32)
    labels = [id2label[int(idx)] for idx in label_ids]
    return labels, stacked


def predict_proba(texts: List[str], max_length: int = 320, batch_size: int = 8, model_dir: str = DEFAULT_DIR) -> np.ndarray:
    _, probs = predict(texts, max_length=max_length, batch_size=batch_size, model_dir=model_dir)
    return probs


__all__ = ["load", "predict", "predict_proba", "DEVICE"]
