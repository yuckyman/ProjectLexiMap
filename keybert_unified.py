#!/usr/bin/env python3
import os
import argparse
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from difflib import SequenceMatcher
import re
import time
import signal
import sys
import nltk
from nltk.corpus import stopwords
from pathlib import Path

# Create directories
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
INTERIM_DIR = RESULTS_DIR / "interim"
INTERIM_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define train and test chapters
TRAIN_CHAPTERS = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
TEST_CHAPTERS = [6, 10, 11, 12]

# Custom stopwords (can be extended)
CUSTOM_STOPWORDS = []

# Debug print function (from evaluator)
def debug_print(message: str, important: bool = False):
    """Print debug message with timestamp and flush immediately."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    prefix = "🔴" if important else "🔹"
    print(f"{prefix} [{timestamp}] {message}", flush=True)

class TimeoutError(Exception):
    """Exception raised when a function execution times out."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")

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

def find_best_match(keyword: str, candidates: List[str], threshold: float = 0.8) -> Tuple[Optional[str], float]:
    """Find the best matching keyword from a list of candidates."""
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        score = keyword_similarity(keyword, candidate)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    return best_match, best_score

def load_chapter(chapter_num: int, verbose: bool = True) -> str:
    """Load a chapter from the textbook folder with optional caching."""
    if verbose:
        debug_print(f"Loading chapter {chapter_num}...")
    
    # Check if cached version exists
    cache_file = CACHE_DIR / f"chapter_{chapter_num}.txt"
    if cache_file.exists():
        try:
            if verbose:
                debug_print(f"Loading chapter {chapter_num} from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if verbose:
                    debug_print(f"Chapter {chapter_num} loaded from cache ({len(content)} chars)")
                return content
        except Exception as e:
            if verbose:
                debug_print(f"Error reading cached chapter {chapter_num}: {e}")
            # Fall back to original loading if cache read fails
    
    try:
        file_path = f'textbook/ch{chapter_num}.txt'
        if verbose:
            debug_print(f"Loading chapter from {file_path}...")
        with open(file_path, 'r') as f:
            content = f.read()
            # Save to cache
            if verbose:
                debug_print(f"Saving chapter {chapter_num} to cache...")
            with open(cache_file, 'w', encoding='utf-8') as cf:
                cf.write(content)
            if verbose:
                debug_print(f"Chapter {chapter_num} loaded ({len(content)} chars)")
            return content
    except Exception as e:
        if verbose:
            debug_print(f"Error loading chapter {chapter_num}: {e}", important=True)
        return ""

def load_index_keywords() -> Dict[int, List[str]]:
    """Load keywords from the index_by_chapter.txt file."""
    debug_print("Starting to load index keywords...", important=True)
    
    # Check if cached version exists
    cache_file = CACHE_DIR / "index_keywords.json"
    if cache_file.exists():
        debug_print("Loading index keywords from cache...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            debug_print(f"Successfully loaded {len(result)} chapters from cache")
            return {int(k): v for k, v in result.items()}
    
    debug_print("Processing index keywords (first run)...", important=True)
    keywords = defaultdict(list)
    current_chapter = None
    
    try:
        debug_print("Opening index_by_chapter.txt...")
        with open('textbook/index_by_chapter.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        debug_print(f"Read {len(lines)} lines from index file")
        
        line_count = 0
        for line in lines:
            line_count += 1
            if line_count % 500 == 0:
                debug_print(f"Processing line {line_count}/{len(lines)}")
                
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Chapter '):
                try:
                    current_chapter = int(line.split()[1])
                    debug_print(f"Found chapter {current_chapter}")
                except (IndexError, ValueError) as e:
                    debug_print(f"Error parsing chapter number from line: {line}")
                    continue
            else:
                # any non-empty line that's not a chapter header is a keyword
                if current_chapter is not None and line:
                    keyword = normalize_keyword(line)
                    if keyword:  # only add if not empty after normalization
                        keywords[current_chapter].append(keyword)
        
        debug_print("Finished processing index file")
        
        # print some stats
        for chapter, words in keywords.items():
            debug_print(f"Chapter {chapter}: {len(words)} keywords")
            if words:
                debug_print(f"Sample keywords: {words[:5]}")
        
        # Save to cache
        debug_print("Saving keywords to cache...")
        result = {str(k): v for k, v in keywords.items()}  # Convert int keys to strings for JSON
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        debug_print("Keywords cached successfully")
        
        # Convert back to int keys for return
        return {int(k): v for k, v in result.items()}
    except Exception as e:
        debug_print(f"Error in load_index_keywords: {str(e)}", important=True)
        raise 

def extract_keywords(text: str, keybert: KeyBERT, chapter_num: Optional[int] = None, 
                    top_n: int = 50, diversity: float = 0.5, 
                    timeout_seconds: int = 120, verbose: bool = True) -> List[Tuple[str, float]]:
    """Extract keywords from text using keybert with caching."""
    if verbose:
        debug_print(f"Extracting keywords for {'chapter ' + str(chapter_num) if chapter_num else 'text'}", important=True)
    
    # If chapter_num is provided, try to load from cache
    if chapter_num is not None:
        cache_file = CACHE_DIR / f"keywords_ch{chapter_num}.json"
        if cache_file.exists():
            if verbose:
                debug_print(f"Loading keywords for chapter {chapter_num} from cache...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if verbose:
                        debug_print(f"Successfully loaded {len(cached_data)} keywords from cache")
                    return [(kw, score) for kw, score in cached_data]
            except Exception as e:
                if verbose:
                    debug_print(f"Error loading cache: {e}")
                # Continue with extraction if cache loading fails
    
    start_time = time.time()
    if verbose:
        debug_print(f"Starting keyword extraction (top_n={top_n}, diversity={diversity})...")
    
    # Get English stopwords and add custom ones
    stop_words = list(stopwords.words('english')) + CUSTOM_STOPWORDS
    
    # Split text into smaller chunks for more focused keyword extraction
    # This helps with long textbook chapters
    chunk_size = 5000  # characters
    overlap = 1000
    chunks = []
    
    if len(text) > chunk_size * 2:  # Only chunk if text is long enough
        if verbose:
            debug_print(f"Text is {len(text)} chars, splitting into chunks...")
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 500:  # Only add reasonably sized chunks
                chunks.append(chunk)
        if verbose:
            debug_print(f"Split into {len(chunks)} chunks")
    else:
        chunks = [text]
    
    try:
        # First try a faster method with MMR (Maximal Marginal Relevance)
        if verbose:
            debug_print("Calling keybert.extract_keywords with MMR method...")
        
        # Set timeout alarm if on a system that supports it
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            has_timeout = True
        except (AttributeError, ValueError):
            # Windows doesn't support SIGALRM
            has_timeout = False
            if verbose:
                debug_print("Timeout not supported on this system")
        
        try:
            keywords = keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),  # Allow up to trigrams
                stop_words=stop_words,
                use_maxsum=False,
                use_mmr=True,
                diversity=diversity,
                top_n=top_n
            )
            
            # Turn off alarm if it was set
            if has_timeout:
                signal.alarm(0)
            
            elapsed = time.time() - start_time
            if verbose:
                debug_print(f"Extracted {len(keywords)} keywords in {elapsed:.2f} seconds", important=True)
                if keywords:
                    debug_print(f"Sample keywords: {keywords[:5]}")
            
            # If we got keywords, save to cache if chapter_num is provided
            if keywords and chapter_num is not None:
                if verbose:
                    debug_print(f"Saving keywords for chapter {chapter_num} to cache...")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    # Convert numpy floats to Python floats for JSON serialization
                    keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                    json.dump(keywords_serializable, f, indent=2)
                if verbose:
                    debug_print("Keywords saved to cache successfully")
            
            return keywords
        
        except TimeoutError:
            # Turn off alarm if it was set
            if has_timeout:
                signal.alarm(0)
            if verbose:
                debug_print("Keyword extraction timed out! Falling back to simpler method...", important=True)
            
            # Fallback to simpler method without maxsum and with fewer keywords
            keywords = keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),  # Reduced n-gram range
                stop_words=stop_words,
                top_n=min(15, top_n),  # Reduce number of keywords for fallback
                use_maxsum=False
            )
        
            elapsed = time.time() - start_time
            if verbose:
                debug_print(f"Extracted {len(keywords)} keywords with fallback in {elapsed:.2f} seconds", important=True)
                if keywords:
                    debug_print(f"Sample fallback keywords: {keywords[:5]}")
            
            # Save to cache if chapter_num is provided
            if chapter_num is not None and keywords:
                if verbose:
                    debug_print(f"Saving fallback keywords for chapter {chapter_num} to cache...")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    # Convert numpy floats to Python floats for JSON serialization
                    keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                    json.dump(keywords_serializable, f, indent=2)
                if verbose:
                    debug_print("Fallback keywords saved to cache successfully")
            
            return keywords
        
    except Exception as e:
        if verbose:
            debug_print(f"Extraction failed: {e}", important=True)
            debug_print("Trying minimal fallback extraction method...")
        
        # Fallback to very simple method
        try:
            keywords = keybert.extract_keywords(
                text[:30000],  # Limit size for minimal fallback
                keyphrase_ngram_range=(1, 1),  # Single words only
                stop_words=stop_words,
                top_n=min(10, top_n)  # Very few keywords
            )
            
            elapsed = time.time() - start_time
            if verbose:
                debug_print(f"Extracted {len(keywords)} keywords with minimal fallback in {elapsed:.2f} seconds", important=True)
                if keywords:
                    debug_print(f"Sample minimal fallback keywords: {keywords[:5]}")
            
            # Save to cache if chapter_num is provided
            if chapter_num is not None and keywords:
                if verbose:
                    debug_print(f"Saving minimal fallback keywords for chapter {chapter_num} to cache...")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    # Convert numpy floats to Python floats for JSON serialization
                    keywords_serializable = [(kw, float(score)) for kw, score in keywords]
                    json.dump(keywords_serializable, f, indent=2)
                if verbose:
                    debug_print("Minimal fallback keywords saved to cache successfully")
            
            return keywords
        except Exception as e2:
            if verbose:
                debug_print(f"Even minimal fallback method failed: {e2}", important=True)
                debug_print("Returning empty keywords list")
            return []

def calculate_metrics(predicted: List[Tuple[str, float]], ground_truth: List[str], 
                     similarity_threshold: float = 0.8, verbose: bool = True) -> Dict[str, Union[float, List]]:
    """Calculate precision, recall, and f1 score using fuzzy matching."""
    if verbose:
        debug_print(f"Calculating metrics: {len(predicted)} predicted vs {len(ground_truth)} ground truth keywords")
    start_time = time.time()
    
    matches = []
    used_truth = set()
    
    # Normalize predicted keywords
    if verbose:
        debug_print("Normalizing predicted keywords...")
    pred_keywords = [(normalize_keyword(kw), score) for kw, score in predicted]
    
    # For each predicted keyword, find the best matching ground truth keyword
    if verbose:
        debug_print("Finding best matches...")
    match_count = 0
    total_predictions = len(pred_keywords)
    
    for i, (pred_kw, pred_score) in enumerate(pred_keywords):
        if verbose and (i % 20 == 0 or i == total_predictions - 1):
            debug_print(f"Matching keyword {i+1}/{total_predictions} ({(i+1)/total_predictions*100:.1f}%)")
            
        best_match, match_score = find_best_match(pred_kw, ground_truth, similarity_threshold)
        if best_match and best_match not in used_truth:
            matches.append((pred_kw, best_match, match_score))
            used_truth.add(best_match)
            match_count += 1
    
    elapsed = time.time() - start_time
    if verbose:
        debug_print(f"Matching complete in {elapsed:.2f} seconds, found {match_count} matches")
    
    true_positives = len(matches)
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    if verbose:
        debug_print(f"\nMatched keywords:")
        for pred, truth, score in matches[:10]:  # Show first 10 matches
            debug_print(f"  {pred} -> {truth} (score: {score:.3f})")
        
        debug_print(f"\nMetrics:")
        debug_print(f"True positives: {true_positives}")
        debug_print(f"False positives: {false_positives}")
        debug_print(f"False negatives: {false_negatives}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if verbose:
        debug_print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": [(p, t, s) for p, t, s in matches]
    }

def evaluate_keywords(predicted: List[Tuple[str, float]], actual: List[Tuple[str, float]], 
                      similarity_threshold: float = 0.7, verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate predicted keywords against actual keywords with multiple metrics.
    Uses fuzzy matching to handle variations in terminology.
    """
    pred_keywords = [kw for kw, _ in predicted]
    actual_keywords = [kw for kw, _ in actual]
    
    if verbose:
        print(f"Evaluating {len(pred_keywords)} predicted vs {len(actual_keywords)} actual keywords")
    
    # Track matches using fuzzy matching
    matches = []
    used_actual = set()
    
    # For each predicted keyword, find best matching actual keyword
    for pred_kw in pred_keywords:
        best_match = None
        best_score = 0
        
        for actual_kw in actual_keywords:
            if actual_kw in used_actual:
                continue
                
            similarity = keyword_similarity(pred_kw, actual_kw)
            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = actual_kw
        
        if best_match:
            matches.append((pred_kw, best_match, best_score))
            used_actual.add(best_match)
    
    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(pred_keywords) - true_positives
    false_negatives = len(actual_keywords) - true_positives
    
    precision = true_positives / len(pred_keywords) if pred_keywords else 0.0
    recall = true_positives / len(actual_keywords) if actual_keywords else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    if verbose and matches:
        print(f"Sample matches:")
        for pred, actual, score in matches[:5]:
            print(f"  - '{pred}' matched with '{actual}' (score: {score:.3f})")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "similarity_threshold": similarity_threshold
    }

def plot_metrics(metrics: Dict[int, Dict[str, float]], output_file: str = "keybert_evaluation.png", verbose: bool = True):
    """Plot evaluation metrics for each chapter."""
    if verbose:
        debug_print("Plotting metrics...", important=True)
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
    if verbose:
        debug_print(f"Saving plot to {output_file}...")
    plt.savefig(output_file)
    plt.close()
    if verbose:
        debug_print("Plot saved successfully")

def train_model(args):
    """Train KeyBERT model on training chapters and output keywords."""
    start_time = time.time()
    debug_print("Starting KeyBERT training process...", important=True)
    debug_print(f"Configuration: model={args.model}, top_n={args.top_n}, diversity={args.diversity}", important=True)
    
    # Initialize KeyBERT
    debug_print(f"Initializing KeyBERT with {args.model}...", important=True)
    try:
        sentence_model = SentenceTransformer(args.model)
        keybert = KeyBERT(model=sentence_model)
        debug_print("KeyBERT initialized successfully")
    except Exception as e:
        debug_print(f"Failed to initialize KeyBERT: {e}", important=True)
        debug_print("Trying to fall back to simpler model...")
        try:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            keybert = KeyBERT(model=sentence_model)
            debug_print("KeyBERT initialized with fallback model")
        except Exception as e2:
            debug_print(f"Fallback initialization also failed: {e2}", important=True)
            return
    
    # Process training chapters
    debug_print(f"Processing {len(TRAIN_CHAPTERS)} training chapters...")
    train_texts = []
    for chapter in TRAIN_CHAPTERS:
        text = load_chapter(chapter)
        if text:
            train_texts.append(text)
    
    # Extract keywords for each training chapter
    debug_print("Extracting keywords from each training chapter...")
    train_keywords_by_chapter = {}
    for i, chapter in enumerate(TRAIN_CHAPTERS):
        text = train_texts[i]
        train_keywords_by_chapter[chapter] = extract_keywords(
            text, 
            keybert, 
            top_n=args.top_n, 
            diversity=args.diversity,
            timeout_seconds=args.timeout
        )
    
    # Combine all unique training keywords
    all_train_keywords = []
    for chapter, keywords in train_keywords_by_chapter.items():
        all_train_keywords.extend(keywords)
    
    # Sort by score and remove duplicates while preserving highest scores
    seen_keywords = set()
    unique_train_keywords = []
    for kw, score in sorted(all_train_keywords, key=lambda x: x[1], reverse=True):
        if kw.lower() not in seen_keywords:
            seen_keywords.add(kw.lower())
            unique_train_keywords.append((kw, score))
    
    # Keep top keywords
    final_top_n = min(75, args.top_n * 3)  # Scale based on per-chapter top_n
    train_keywords = unique_train_keywords[:final_top_n]
    
    debug_print("\nTop training keywords:")
    for kw, score in train_keywords[:10]:
        debug_print(f"  - {kw}: {score:.4f}")
    
    # Evaluate on test chapters if requested
    test_results = {}
    if not args.skip_evaluation:
        debug_print(f"\nEvaluating on {len(TEST_CHAPTERS)} test chapters...")
        for chapter in TEST_CHAPTERS:
            test_text = load_chapter(chapter)
            if test_text:
                debug_print(f"\nProcessing test chapter {chapter}...")
                test_keywords = extract_keywords(
                    test_text, 
                    keybert, 
                    top_n=args.top_n, 
                    diversity=args.diversity,
                    timeout_seconds=args.timeout
                )
                
                # Evaluate against training keywords
                metrics = evaluate_keywords(
                    test_keywords, 
                    train_keywords, 
                    similarity_threshold=args.similarity_threshold
                )
                
                debug_print(f"Results for chapter {chapter}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        debug_print(f"  - {metric}: {value:.4f}")
                
                test_results[f"chapter_{chapter}"] = {
                    "keywords": [(kw, float(score)) for kw, score in test_keywords],
                    "metrics": metrics
                }
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_name_short = args.model.split('/')[-1]
    output_file = RESULTS_DIR / f"keybert_{model_name_short}_train_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "train_keywords": [(kw, float(score)) for kw, score in train_keywords],
            "test_results": test_results,
            "parameters": {
                "model": args.model,
                "top_n": args.top_n,
                "diversity": args.diversity,
                "similarity_threshold": args.similarity_threshold
            }
        }, f, indent=2)
    
    debug_print(f"\nResults saved to {output_file}")
    
    # Report execution time
    total_time = time.time() - start_time
    debug_print(f"Total execution time: {total_time:.2f} seconds", important=True)
    debug_print("Training process complete!", important=True)

def evaluate_model(args):
    """Evaluate KeyBERT model against textbook index."""
    overall_start_time = time.time()
    debug_print("Starting evaluation process...", important=True)
    debug_print(f"Configuration: model={args.model}, top_n={args.top_n}, diversity={args.diversity}, "
                f"similarity_threshold={args.similarity_threshold}, timeout={args.timeout}", important=True)
    
    # Clear cache if requested
    if args.clear_cache:
        debug_print("Clearing cache as requested...", important=True)
        for cache_file in CACHE_DIR.glob("*"):
            try:
                cache_file.unlink()
                debug_print(f"Deleted {cache_file}")
            except Exception as e:
                debug_print(f"Error deleting {cache_file}: {e}")
    
    # Clean up old results if requested
    if args.clean_results:
        debug_print("Cleaning up old results as requested...", important=True)
        # Clean interim results
        for old_file in INTERIM_DIR.glob("*"):
            try:
                old_file.unlink()
                debug_print(f"Deleted {old_file}")
            except Exception as e:
                debug_print(f"Error deleting {old_file}: {e}")
        
        # Clean main results with same model name
        model_name_short = args.model.split('/')[-1]
        for old_file in RESULTS_DIR.glob(f"keybert_{model_name_short}_*.json"):
            try:
                old_file.unlink()
                debug_print(f"Deleted {old_file}")
            except Exception as e:
                debug_print(f"Error deleting {old_file}: {e}")
    
    # Load ground truth keywords from index
    try:
        chapter_keywords = load_index_keywords()
        debug_print(f"Loaded keywords for {len(chapter_keywords)} chapters")
    except Exception as e:
        debug_print(f"Failed to load index keywords: {e}", important=True)
        return
    
    # Initialize keybert
    debug_print(f"Initializing keybert model with {args.model}...", important=True)
    try:
        sentence_model = SentenceTransformer(args.model)
        keybert = KeyBERT(model=sentence_model)
        debug_print("KeyBERT initialized successfully")
    except Exception as e:
        debug_print(f"Failed to initialize KeyBERT: {e}", important=True)
        debug_print("Trying to fall back to simpler model...")
        try:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            keybert = KeyBERT(model=sentence_model)
            debug_print("KeyBERT initialized with fallback model")
        except Exception as e2:
            debug_print(f"Fallback initialization also failed: {e2}", important=True)
            return
    
    # Evaluate each chapter
    results = {}
    metrics = {}
    
    # Determine chapters to evaluate
    chapters_to_evaluate = args.chapters if args.chapters else sorted(chapter_keywords.keys())
    total_chapters = len(chapters_to_evaluate)
    processed = 0
    
    for chapter_num in chapters_to_evaluate:
        chapter_start_time = time.time()
        processed += 1
        debug_print(f"Processing chapter {chapter_num} ({processed}/{total_chapters})...", important=True)
        
        # Get chapter text
        try:
            text = load_chapter(chapter_num)
            debug_print(f"Chapter {chapter_num} text length: {len(text)} characters")
            if not text:
                debug_print(f"Skipping chapter {chapter_num} - no text available", important=True)
                continue
        except Exception as e:
            debug_print(f"Error loading chapter {chapter_num}: {e}", important=True)
            continue
            
        # Extract keywords with caching
        try:
            debug_print(f"Extracting keywords for chapter {chapter_num}...")
            keywords = extract_keywords(
                text, 
                keybert, 
                chapter_num=chapter_num,
                top_n=args.top_n,
                diversity=args.diversity,
                timeout_seconds=args.timeout
            )
            debug_print(f"Extracted {len(keywords)} keywords for chapter {chapter_num}")
        except Exception as e:
            debug_print(f"Error extracting keywords for chapter {chapter_num}: {e}", important=True)
            continue
        
        # Calculate metrics
        try:
            debug_print(f"Calculating metrics for chapter {chapter_num}...")
            chapter_metrics = calculate_metrics(
                keywords, 
                chapter_keywords[chapter_num],
                similarity_threshold=args.similarity_threshold
            )
            metrics[chapter_num] = chapter_metrics
            debug_print(f"Chapter {chapter_num} metrics: {chapter_metrics['precision']:.3f} precision, {chapter_metrics['recall']:.3f} recall, {chapter_metrics['f1']:.3f} F1")
        except Exception as e:
            debug_print(f"Error calculating metrics for chapter {chapter_num}: {e}", important=True)
            continue
        
        # Store results
        results[f"chapter_{chapter_num}"] = {
            "extracted_keywords": [(kw, float(score)) for kw, score in keywords],  # convert numpy float to python float
            "ground_truth": chapter_keywords[chapter_num],
            "metrics": {k: v for k, v in chapter_metrics.items() if k != 'matches'}  # exclude matches from JSON
        }
        
        chapter_elapsed = time.time() - chapter_start_time
        debug_print(f"Chapter {chapter_num} processed in {chapter_elapsed:.2f} seconds", important=True)
        
        # Save interim results every few chapters
        if not args.no_interim and (processed % 3 == 0 or processed == total_chapters):
            debug_print(f"Saving interim results after processing {processed}/{total_chapters} chapters...")
            try:
                model_name_short = args.model.split('/')[-1]
                interim_file = INTERIM_DIR / f"keybert_{model_name_short}_interim_{processed}of{total_chapters}.json"
                with open(interim_file, "w") as f:
                    json.dump(results, f, indent=2)
                debug_print(f"Interim results saved to {interim_file}")
            except Exception as e:
                debug_print(f"Error saving interim results: {e}", important=True)
    
    # Save detailed results
    debug_print("Saving final evaluation results...", important=True)
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        model_name_short = args.model.split('/')[-1]
        output_file = RESULTS_DIR / f"keybert_{model_name_short}_results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "top_n": args.top_n,
                    "diversity": args.diversity,
                    "similarity_threshold": args.similarity_threshold,
                    "timeout": args.timeout
                },
                "results": results
            }, f, indent=2)
        debug_print(f"Evaluation results saved to {output_file}")
    except Exception as e:
        debug_print(f"Error saving final results: {e}", important=True)
    
    # Plot metrics
    try:
        plot_file = RESULTS_DIR / f"keybert_{model_name_short}_evaluation_{timestamp}.png"
        plot_metrics(metrics, output_file=str(plot_file))
        debug_print(f"Metrics plot saved to {plot_file}")
    except Exception as e:
        debug_print(f"Error plotting metrics: {e}", important=True)
    
    # Print average metrics
    debug_print("Calculating average metrics...", important=True)
    avg_metrics = defaultdict(float)
    for m in metrics.values():
        for k, v in m.items():
            if isinstance(v, (int, float)):  # only average numeric metrics
                avg_metrics[k] += v
    
    for k in avg_metrics:
        avg_metrics[k] /= len(metrics)
    
    debug_print("\nAverage metrics across all chapters:", important=True)
    debug_print(f"Precision: {avg_metrics['precision']:.3f}")
    debug_print(f"Recall: {avg_metrics['recall']:.3f}")
    debug_print(f"F1 score: {avg_metrics['f1']:.3f}")
    debug_print(f"Avg true positives: {avg_metrics['true_positives']:.1f}")
    debug_print(f"Avg false positives: {avg_metrics['false_positives']:.1f}")
    debug_print(f"Avg false negatives: {avg_metrics['false_negatives']:.1f}")
    
    total_time = time.time() - overall_start_time
    debug_print(f"\nTotal execution time: {total_time:.2f} seconds", important=True)
    debug_print("Evaluation complete!", important=True)

def main():
    # Create top-level parser
    parser = argparse.ArgumentParser(description='Unified KeyBERT trainer and evaluator for keyword extraction')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    subparsers.required = True
    
    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('--model', type=str, default='all-mpnet-base-v2',
                        help='Sentence transformer model to use (default: all-mpnet-base-v2)')
    common_args.add_argument('--top_n', type=int, default=50,
                        help='Number of keywords to extract per chapter (default: 50)')
    common_args.add_argument('--diversity', type=float, default=0.5,
                        help='Diversity factor for keyword selection (0-1) (default: 0.5)')
    common_args.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='Similarity threshold for matching keywords (default: 0.8)')
    common_args.add_argument('--timeout', type=int, default=120,
                        help='Timeout in seconds for keyword extraction (default: 120)')
    common_args.add_argument('--clear_cache', action='store_true',
                        help='Clear the cache before running')
    
    # Create parser for the train command
    train_parser = subparsers.add_parser('train', parents=[common_args],
                                     help='Train KeyBERT model on training chapters')
    train_parser.add_argument('--skip_evaluation', action='store_true',
                          help='Skip evaluation on test chapters')
    train_parser.add_argument('--clean_results', action='store_true',
                          help='Clean up old training results before running')
    
    # Create parser for the evaluate command
    eval_parser = subparsers.add_parser('evaluate', parents=[common_args],
                                    help='Evaluate KeyBERT model against textbook index')
    eval_parser.add_argument('--chapters', type=int, nargs='+',
                         help='Specific chapters to evaluate (default: all chapters)')
    eval_parser.add_argument('--no_interim', action='store_true',
                         help='Disable saving interim results')
    eval_parser.add_argument('--clean_results', action='store_true',
                         help='Clean up old evaluation results before running')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        debug_print(f"CRITICAL ERROR: {str(e)}", important=True)
        import traceback
        debug_print(traceback.format_exc(), important=True) 