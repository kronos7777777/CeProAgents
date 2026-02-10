# src/commons/sem_embed.py
import os
import numpy as np

_BACKEND = None
_MODEL = None

def _ensure_backend():
    global _BACKEND, _MODEL
    if _BACKEND is not None:
        return _BACKEND, _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-mpnet-base-v2")
        _MODEL = SentenceTransformer(model_name)
        _BACKEND = "sbert"
    except Exception:
        # if no sbert then TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        _MODEL = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        _BACKEND = "tfidf"
    return _BACKEND, _MODEL

def embed_texts(texts):
    """
    Input: List[str]; Output: (N, D) float32, L2 Normalized
    """
    backend, model = _ensure_backend()
    if backend == "sbert":
        vec = model.encode(
            texts,
            batch_size=int(os.getenv("EMBED_BATCH", "256")),
            show_progress_bar=False,
            normalize_embeddings=True,
            device= "cuda"
        )
        return np.asarray(vec, dtype=np.float32)
    else:
        X = model.fit_transform(texts)
        norms = np.sqrt((X.multiply(X)).sum(axis=1)).A.ravel() + 1e-12
        X = X.multiply(1.0 / norms[:, None])
        return X.toarray().astype(np.float32)

def cosine_matrix(A, B=None):
    if B is None:
        B = A
    return A @ B.T
