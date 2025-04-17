#%% imports and utilities
import json
import re
import time
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import nltk
from nltk.corpus import stopwords
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#%% constants and paths
TRAIN_CHAPTERS = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
TEST_CHAPTERS  = [6, 10, 11, 12]
CUSTOM_STOPWORDS = [
    'chapter', 'figure', 'section', 'example', 'et', 'al', 'et al',
    'ii', 'iii', 'also', 'thus', 'however', 'moreover', 'therefore',
    'hence', 'e.g.', 'i.e.', 'vs', 'etc', 'fig', 'mean', 'note',
    'defined', 'definition', 'approach', 'way', 'www', 'org'
]
EVAL_SIMILARITY_THRESHOLD = 0.75
TEXTBOOK_DIR = Path("textbook")
RESULTS_DIR  = Path("results")

#%% nltk setup
# download stopwords; punkt is not needed due to custom splitter below
nltk.download('stopwords', quiet=True)


#%% helper functions
@lru_cache(maxsize=1000)
def get_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Return and cache embedding vector for `text` using `model`.
    """
    return model.encode([text])[0]


def split_sentences(text: str) -> List[str]:
    """
    Simple regex-based sentence splitter to avoid NLTK punkt lookup issues.
    Splits text at punctuation boundaries (., !, ?).
    """
    # split on punctuation followed by whitespace(s)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def load_chapter(chapter_num: int) -> str:
    """
    Load text content of a chapter file.
    Returns empty string on error.
    """
    path = TEXTBOOK_DIR / f"ch{chapter_num}.txt"
    try:
        content = path.read_text(encoding="utf-8")
        print(f"Loaded chapter {chapter_num} ({len(content)} chars)")
        return content
    except Exception as e:
        print(f"Error loading chapter {chapter_num}: {e}")
        return ""


def load_index_keywords(index_file: Path = TEXTBOOK_DIR / "index_by_chapter.txt"
                       ) -> Dict[int, List[str]]:
    """
    Parse ground-truth index file into a mapping: chapter -> list of keywords.
    """
    chapters = defaultdict(list)
    current = None
    for line in index_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("chapter"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                current = int(parts[1])
        elif current is not None:
            chapters[current].append(line)
    return chapters

def is_valid_keyword(kw: str) -> bool:
    """
    Validate that `kw` is a reasonable keyword (not too short or math-like).
    Allows multi-word terms and common technical characters.
    """
    kw = kw.strip()
    if not kw or len(kw) < 3:
        return False
    if len(kw.split()) > 1:
        return True
    # allow alphanumeric, dashes, parentheses, ampersands, plus
    return bool(re.match(r'^[A-Za-z0-9\-\(\)&\+]+$', kw))


#%% keyword extraction
def extract_hybrid_keywords(
    text: str,
    keybert_model: KeyBERT,
    top_n: int = 75,
    diversity: float = 0.6
) -> List[Tuple[str, float]]:
    """
    Extract top_n keywords by combining KeyBERT (with MMR) and TF-IDF.
    1) Split `text` into chunks for coherence.
    2) Use KeyBERT for semantic extraction.
    3) Use TF-IDF for term-frequency scoring.
    4) Merge and rank results, filtering invalid terms.
    """
    # Combine standard and custom stop words
    stop_words_list = list(stopwords.words('english')) + CUSTOM_STOPWORDS

    # split into manageable chunks
    sentences = split_sentences(text)
    chunk_size = 5000
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) <= chunk_size:
            buf += s + " "
        else:
            chunks.append(buf.strip())
            buf = s + " "
    if buf:
        chunks.append(buf.strip())

    # KeyBERT extraction using combined stop words
    neural = keybert_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=stop_words_list,
        use_mmr=True,
        diversity=diversity,
        top_n=top_n
    )
    neural = [(kw, score) for kw, score in neural if is_valid_keyword(kw)]

    # TF-IDF extraction using combined stop words
    vectorizer = TfidfVectorizer(stop_words=stop_words_list, min_df=1)
    tfidf_mat = vectorizer.fit_transform(chunks)
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf_mat.sum(axis=0)).ravel()
    tfidf_terms = sorted(
        [(terms[i], float(scores[i])) for i in range(len(terms))],
        key=lambda x: x[1], reverse=True
    )[:top_n]
    tfidf_terms = [(kw, score) for kw, score in tfidf_terms if is_valid_keyword(kw)]

    # merge scores from both sources
    combined = defaultdict(float)
    for kw, score in neural + tfidf_terms:
        combined[kw] += score

    # pick top_n
    final = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return final


#%% evaluation and plotting
def evaluate_keywords(
    predicted: List[Tuple[str, float]],
    actual: List[str],
    model: SentenceTransformer,
    threshold: float = EVAL_SIMILARITY_THRESHOLD
) -> Dict[str, float]:
    """
    Match predicted vs actual keywords by embedding similarity.
    Compute precision, recall, and F1 score.
    """
    preds = [kw for kw, _ in predicted]
    used, matches = set(), []

    # for each predicted, find best actual match above threshold
    for p in preds:
        embp = get_embedding(p, model)
        best_sim, best_act = 0.0, None
        for a in actual:
            if a in used:
                continue
            sim = cosine_similarity([embp], [get_embedding(a, model)])[0][0]
            if sim >= threshold and sim > best_sim:
                best_sim, best_act = sim, a
        if best_act:
            matches.append((p, best_act, best_sim))
            used.add(best_act)

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(actual) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    # cast to native floats so JSON serialization won't choke on numpy types
    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1_score":  float(f1),
        # ensure each similarity score is a Python float
        "matches":  [(p, a, float(s)) for p, a, s in matches]
    }


def plot_metrics(
    metrics: Dict[int, Dict[str, float]],
    output_file: str
):
    """
    Create and save a bar chart of precision, recall, F1 for each chapter.
    """
    chapters = sorted(metrics)
    precision = [metrics[c]["precision"] for c in chapters]
    recall    = [metrics[c]["recall"]    for c in chapters]
    f1        = [metrics[c]["f1_score"]  for c in chapters]

    x = np.arange(len(chapters))
    plt.figure(figsize=(10, 6))
    plt.bar(x-0.2, precision, width=0.2, label="Precision")
    plt.bar(x,    recall,    width=0.2, label="Recall")
    plt.bar(x+0.2, f1,       width=0.2, label="F1 Score")
    plt.xticks(x, chapters)
    plt.xlabel("Chapter")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved metrics plot: {output_file}")


#%% main execution
def main():
    """
    Orchestrate:
      - Model loading
      - Keyword extraction on train/test
      - Evaluation against ground truth
      - Saving results and plots
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Loading embedding model...")
    try:
        embed_model = SentenceTransformer("models/mpnet_textbook_tuned")
    except Exception:
        print("Fallback to base 'all-mpnet-base-v2'")
        embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("Initializing KeyBERT model...")
    kw_model = KeyBERT(model=embed_model)

    print("Extracting training keywords...")
    train_texts = [load_chapter(c) for c in TRAIN_CHAPTERS]
    train_kws   = {c: extract_hybrid_keywords(t, kw_model) for c, t in zip(TRAIN_CHAPTERS, train_texts) if t}

    print("Loading ground truth index...")
    index = load_index_keywords()

    print("Evaluating on test chapters...")
    results = {}
    for chap in TEST_CHAPTERS:
        text = load_chapter(chap)
        if not text:
            continue
        print(f" - Chapter {chap}")
        pred   = extract_hybrid_keywords(text, kw_model)
        actual = index.get(chap, [])
        results[chap] = evaluate_keywords(pred, actual, embed_model)

    # save evaluation JSON (all values are now serializable)
    timestamp = int(time.time())
    out_json = RESULTS_DIR / f"evaluation_{timestamp}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation JSON to {out_json}")

    # plot metrics
    plot_metrics(results, str(RESULTS_DIR / f"metrics_{timestamp}.png"))


if __name__ == "__main__":
    main()
