import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from difflib import SequenceMatcher
import re

def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword by removing special characters and converting to lowercase."""
    # Remove special characters and extra whitespace
    keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    return keyword

def keyword_similarity(kw1: str, kw2: str) -> float:
    """Calculate similarity between two keywords."""
    # Normalize keywords
    kw1 = normalize_keyword(kw1)
    kw2 = normalize_keyword(kw2)
    
    # If exact match after normalization
    if kw1 == kw2:
        return 1.0
    
    # Check if one is contained in the other
    if kw1 in kw2 or kw2 in kw1:
        return 0.9
    
    # Use sequence matcher for fuzzy matching
    return SequenceMatcher(None, kw1, kw2).ratio()

def find_best_match(keyword: str, candidates: List[str], threshold: float = 0.8) -> Tuple[str, float]:
    """Find the best matching keyword from a list of candidates."""
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        score = keyword_similarity(keyword, candidate)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    return best_match, best_score

def load_index_keywords() -> Dict[int, List[str]]:
    """Load keywords from the index_by_chapter.txt file."""
    keywords = defaultdict(list)
    current_chapter = None
    
    with open('textbook/index_by_chapter.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Chapter '):
            try:
                current_chapter = int(line.split()[1])
                print(f"found chapter {current_chapter}")
            except (IndexError, ValueError) as e:
                print(f"error parsing chapter number from line: {line}")
                continue
        else:
            # any non-empty line that's not a chapter header is a keyword
            if current_chapter is not None and line:
                keyword = normalize_keyword(line)
                if keyword:  # only add if not empty after normalization
                    keywords[current_chapter].append(keyword)
    
    # print some stats
    for chapter, words in keywords.items():
        print(f"chapter {chapter}: {len(words)} keywords")
        if words:
            print(f"sample keywords: {words[:5]}")
    
    return dict(keywords)

def load_chapter(chapter_num: int) -> str:
    """load a chapter from the textbook folder."""
    try:
        with open(f'textbook/ch{chapter_num}.txt', 'r') as f:
            return f.read()
    except Exception as e:
        print(f"error loading chapter {chapter_num}: {e}")
        return ""

def extract_keywords(text: str, keybert: KeyBERT, top_n: int = 50) -> List[Tuple[str, float]]:
    """extract keywords from text using keybert."""
    # Extract single words and phrases
    keywords = keybert.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=top_n,
        use_maxsum=True,
        diversity=0.7
    )
    
    print(f"extracted {len(keywords)} keywords")
    print(f"sample keywords: {keywords[:5]}")
    return keywords

def calculate_metrics(predicted: List[Tuple[str, float]], ground_truth: List[str], 
                     similarity_threshold: float = 0.8) -> Dict[str, float]:
    """calculate precision, recall, and f1 score using fuzzy matching."""
    matches = []
    used_truth = set()
    
    # Normalize predicted keywords
    pred_keywords = [(normalize_keyword(kw), score) for kw, score in predicted]
    
    # For each predicted keyword, find the best matching ground truth keyword
    for pred_kw, pred_score in pred_keywords:
        best_match, match_score = find_best_match(pred_kw, ground_truth, similarity_threshold)
        if best_match and best_match not in used_truth:
            matches.append((pred_kw, best_match, match_score))
            used_truth.add(best_match)
    
    true_positives = len(matches)
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    print(f"\nMatched keywords:")
    for pred, truth, score in matches[:10]:  # Show first 10 matches
        print(f"  {pred} -> {truth} (score: {score:.3f})")
    
    print(f"\nMetrics:")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": [(p, t, s) for p, t, s in matches]
    }

def plot_metrics(metrics: Dict[int, Dict[str, float]], output_file: str = "keybert_evaluation.png"):
    """plot evaluation metrics for each chapter."""
    chapters = sorted(metrics.keys())
    precisions = [metrics[ch]["precision"] for ch in chapters]
    recalls = [metrics[ch]["recall"] for ch in chapters]
    f1_scores = [metrics[ch]["f1"] for ch in chapters]
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(chapters))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Chapter')
    plt.ylabel('Score')
    plt.title('KeyBERT Performance by Chapter')
    plt.xticks(x, chapters)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # load ground truth keywords from index
    chapter_keywords = load_index_keywords()
    
    # initialize keybert
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    keybert = KeyBERT(model=sentence_model)
    
    # evaluate each chapter
    results = {}
    metrics = {}
    
    for chapter_num in sorted(chapter_keywords.keys()):
        print(f"\nprocessing chapter {chapter_num}...")
        
        # get chapter text
        text = load_chapter(chapter_num)
        if not text:
            continue
            
        # extract keywords
        keywords = extract_keywords(text, keybert)
        
        # calculate metrics
        chapter_metrics = calculate_metrics(keywords, chapter_keywords[chapter_num])
        metrics[chapter_num] = chapter_metrics
        
        # store results
        results[f"chapter_{chapter_num}"] = {
            "extracted_keywords": [(kw, float(score)) for kw, score in keywords],  # convert numpy float to python float
            "ground_truth": chapter_keywords[chapter_num],
            "metrics": chapter_metrics
        }
    
    # save detailed results
    with open("keybert_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # plot metrics
    plot_metrics(metrics)
    
    # print average metrics
    avg_metrics = defaultdict(float)
    for m in metrics.values():
        for k, v in m.items():
            if isinstance(v, (int, float)):  # only average numeric metrics
                avg_metrics[k] += v
    
    for k in avg_metrics:
        avg_metrics[k] /= len(metrics)
    
    print("\naverage metrics across all chapters:")
    print(f"precision: {avg_metrics['precision']:.3f}")
    print(f"recall: {avg_metrics['recall']:.3f}")
    print(f"f1 score: {avg_metrics['f1']:.3f}")
    print(f"avg true positives: {avg_metrics['true_positives']:.1f}")
    print(f"avg false positives: {avg_metrics['false_positives']:.1f}")
    print(f"avg false negatives: {avg_metrics['false_negatives']:.1f}")

if __name__ == "__main__":
    main() 