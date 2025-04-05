import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple
import zipfile
import time
import nltk
from nltk.corpus import stopwords
import re
from difflib import SequenceMatcher
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache

# Create results directory if it doesn't exist
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# define train and test chapters
TRAIN_CHAPTERS = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
TEST_CHAPTERS = [6, 10, 11, 12]

# Using only NLTK's stopwords
CUSTOM_STOPWORDS = ['chapter', 'figure', 'section', 'example', 'page', 'table', 'see', 'shown', 'thus', 'since', 'however', 'therefore', 'might', 'may', 'could', 'would', 'should', 'can', 'one', 'two', 'three', 'first', 'second', 'third', 'use', 'using', 'used', 'uses', 'following', 'show', 'consider', 'case', 'different', 'given', 'set', 'called', 'define', 'contains', 'describes', 'way', 'mean', 'note', 'defined', 'definition', 'approach']

@lru_cache(maxsize=1000)
def get_embedding(text, model):
    """Cache embeddings to avoid recomputing them."""
    return model.encode([text])[0]

def load_chapter(chapter_num: int) -> str:
    """load a chapter from the textbook folder."""
    try:
        print(f"Loading chapter {chapter_num}...")
        with open(f'textbook/ch{chapter_num}.txt', 'r') as f:
            content = f.read()
            print(f"Chapter {chapter_num} loaded: {len(content)} chars")
            return content
    except Exception as e:
        print(f"Error loading chapter {chapter_num}: {e}")
        return ""

def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword by removing special characters and converting to lowercase."""
    # First clean any soft hyphens before normalization
    keyword = keyword.replace("‐ ", "").replace("‐", "")
    
    # Remove special characters and extra whitespace
    keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    return keyword

def is_valid_keyword(keyword: str) -> bool:
    """
    Check if a keyword is valid using regex pattern.
    Includes common characters for technical terms but excludes mathematical expressions.
    """
    # Skip empty keywords
    if not keyword or len(keyword.strip()) == 0:
        return False
        
    # For multi-word keywords, check each word
    words = keyword.split()
    if len(words) > 1:
        # Allow multi-word terms even if individual parts wouldn't be valid alone
        return True
    
    # Special cases for known technical terms with + signs
    special_cases = ["C++", "C&C++", "A+B"]
    if keyword in special_cases:
        return True
    
    # Check for common mathematical expressions that should be excluded
    math_patterns = [
        r'=',                # equations
        r'[*/÷]',            # arithmetic operators (excluding + which can be in C++)
        r'\s[<>]\s',         # inequality signs with spaces
        r'≠|≤|≥|→|←',        # special math symbols
        r'∀|∃|∈|⊂|⊃|∪|∩',    # set theory symbols
        r'\$\\',             # LaTeX marker
        r'\\frac',           # LaTeX fraction
        r'^[0-9+*/-]+$',     # pure arithmetic expressions
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, keyword):
            # Skip the check for special cases
            if keyword in special_cases:
                continue
            return False
    
    # For single words, check if they match our pattern for normal terms
    # Allow common characters used in technical terms, including commas
    return bool(re.match(r'^[A-Za-z0-9\-\'_\\(),\[\]&+.:^]+$', keyword))

def extract_keywords(text: str, keybert: KeyBERT, top_n: int = 30, 
                    diversity: float = 0.5, min_df: int = 1) -> List[Tuple[str, float]]:
    """
    Extract keywords from text using keybert with parameters optimized for educational content.
    
    Args:
        text: Input text
        keybert: KeyBERT model
        top_n: Number of keywords to extract
        diversity: Diversity of keywords (0-1)
        min_df: Minimum document frequency for candidates
        
    Returns:
        List of tuples (keyword, score)
    """
    print(f"Extracting keywords (top_n={top_n}, diversity={diversity})...")
    start_time = time.time()
    
    # Get English stopwords and add custom ones
    stop_words = list(stopwords.words('english')) + CUSTOM_STOPWORDS
    
    # Split text into smaller chunks for more focused keyword extraction
    # This helps with long textbook chapters
    chunk_size = 5000  # characters
    overlap = 1000
    chunks = []
    
    if len(text) > chunk_size * 2:  # Only chunk if text is long enough
        print(f"Text is {len(text)} chars, splitting into chunks...")
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 500:  # Only add reasonably sized chunks
                chunks.append(chunk)
        print(f"Split into {len(chunks)} chunks")
    else:
        chunks = [text]
    
    # Extract keywords from each chunk
    all_keywords = []
    
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        try:
            # Extract keywords with MMR (Maximal Marginal Relevance)
            keywords = keybert.extract_keywords(
                chunk,
                keyphrase_ngram_range=(1, 2),  # Allow up to trigrams
                stop_words=stop_words,
                use_maxsum=False,  # Use MMR instead for better extraction with mpnet
                use_mmr=True,
                diversity=diversity,
                top_n=min(30, top_n),  # Extract more from each chunk
                min_df=min_df
            )
            
            # Filter out invalid keywords (math formulae, etc.)
            filtered_keywords = []
            for kw, score in keywords:
                if is_valid_keyword(kw):
                    filtered_keywords.append((kw, score))
                else:
                    print(f"Filtered out invalid keyword: {kw}")
            
            if len(filtered_keywords) < len(keywords):
                print(f"Filtered out {len(keywords) - len(filtered_keywords)} invalid keywords")
            
            keywords = filtered_keywords
            
            if len(keywords) > 0:
                all_keywords.extend(keywords)
            
        except Exception as e:
            print(f"Error extracting keywords from chunk {i+1}: {e}")
    
    # If we have keywords from multiple chunks, combine and deduplicate them
    final_keywords = []
    if len(chunks) > 1 and all_keywords:
        # Sort by score and remove duplicates
        seen_keywords = set()
        for kw, score in sorted(all_keywords, key=lambda x: x[1], reverse=True):
            if kw.lower() not in seen_keywords:
                seen_keywords.add(kw.lower())
                final_keywords.append((kw, score))
        
        # Take top keywords
        final_keywords = final_keywords[:top_n]
    else:
        final_keywords = all_keywords[:top_n]
    
    elapsed = time.time() - start_time
    print(f"Extracted {len(final_keywords)} keywords in {elapsed:.2f} seconds")
    
    if final_keywords:
        print("Sample keywords:")
        for kw, score in final_keywords[:5]:
            print(f"  - {kw}: {score:.4f}")
    elif len(text) > 100:
        # If extraction failed completely, try a simpler approach as fallback
        print("Keyword extraction produced no results. Trying fallback method...")
        try:
            keywords = keybert.extract_keywords(
                text[:30000],  # Limit size for fallback
                keyphrase_ngram_range=(1, 2),  # Simpler extraction
                stop_words=stop_words,
                top_n=top_n
            )
            final_keywords = keywords
            print(f"Fallback extracted {len(final_keywords)} keywords")
        except Exception as e2:
            print(f"Fallback extraction failed too: {e2}")
    
    return final_keywords

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

def embedding_similarity(kw1, kw2, model):
    # Get cached embeddings for both keywords
    emb1 = get_embedding(kw1, model)
    emb2 = get_embedding(kw2, model)
    
    # Calculate cosine similarity
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim)

def ngram_match(kw1: str, kw2: str) -> float:
    """Calculate similarity based on shared ngrams."""
    # Normalize keywords
    kw1 = normalize_keyword(kw1)
    kw2 = normalize_keyword(kw2)
    
    # Split into words
    words1 = kw1.split()
    words2 = kw2.split()
    
    # Calculate intersection of words
    common = set(words1) & set(words2)
    
    if not common:  # No words in common
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    jaccard = len(common) / len(set(words1) | set(words2))
    
    # Calculate word overlap ratio
    overlap_ratio = len(common) / min(len(words1), len(words2))
    
    # Weighted combination of both similarities
    return 0.5 * jaccard + 0.5 * overlap_ratio

def hierarchical_match(pred_kw: str, actual_kw: str) -> float:
    """Use a hierarchical approach to matching keywords."""
    norm_pred = normalize_keyword(pred_kw)
    norm_actual = normalize_keyword(actual_kw)
    
    # Exact match
    if norm_pred == norm_actual:
        return 1.0
    
    # Perfect containment - one is exactly inside the other
    if norm_pred in norm_actual:
        # If predicted is a substring of actual but nearly as long, score it higher
        length_ratio = len(norm_pred) / len(norm_actual)
        if length_ratio > 0.7:  # Predicted captures most of the actual keyword
            return 0.9
        return 0.8 * length_ratio  # Lower score for smaller substrings
    
    if norm_actual in norm_pred:
        # If actual is a substring of predicted but nearly as long, score it higher
        length_ratio = len(norm_actual) / len(norm_pred)
        if length_ratio > 0.7:  # Actual captures most of the predicted keyword
            return 0.9
        return 0.8 * length_ratio  # Lower score for smaller substrings
    
    # Word overlap using ngram matching
    ngram_score = ngram_match(norm_pred, norm_actual)
    if ngram_score > 0.6:  # Significant word overlap
        return 0.7 + (0.2 * ngram_score)  # Scale between 0.7-0.9
    
    # Fall back to traditional string similarity for low overlap cases
    return keyword_similarity(norm_pred, norm_actual) * 0.7  # Scale down string similarity

def advanced_match_keywords(pred_kw: str, actual_kws: List[str], model, similarity_threshold: float = 0.7) -> Tuple[str, float]:
    """
    Match a predicted keyword against a list of actual keywords using multiple techniques.
    
    Args:
        pred_kw: Predicted keyword to match
        actual_kws: List of actual keywords to match against
        model: Sentence transformer model for embedding similarity
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (best_match, similarity_score)
    """
    best_match = None
    best_score = 0
    
    for actual_kw in actual_kws:
        # Try hierarchical matching first
        h_score = hierarchical_match(pred_kw, actual_kw)
        
        # If we get a strong hierarchical match
        if h_score >= similarity_threshold:
            if h_score > best_score:
                best_score = h_score
                best_match = actual_kw
            continue
        
        # Try embedding similarity if hierarchical matching wasn't strong enough
        if h_score < similarity_threshold:
            e_score = embedding_similarity(pred_kw, actual_kw, model)
            
            # Use the better of the two scores
            sim_score = max(h_score, e_score)
            
            if sim_score > best_score and sim_score >= similarity_threshold:
                best_score = sim_score
                best_match = actual_kw
    
    return best_match, best_score

def evaluate_keywords(predicted: List[Tuple[str, float]], actual: List[Tuple[str, float]], 
                      model, similarity_threshold: float = 0.7) -> Dict[str, float]:
    """
    Evaluate predicted keywords against actual keywords with multiple metrics.
    Uses hierarchical matching and embedding similarity to handle variations.
    """
    pred_keywords = [kw for kw, _ in predicted]
    actual_keywords = [kw for kw, _ in actual]
    
    print(f"Evaluating {len(pred_keywords)} predicted vs {len(actual_keywords)} actual keywords")
    
    # Track matches using advanced matching
    matches = []
    used_actual = set()
    
    # For each predicted keyword, find best matching actual keyword
    for pred_kw in pred_keywords:
        # Skip already processed predictions (exact duplicates)
        norm_pred = normalize_keyword(pred_kw)
        
        # Find best match using advanced matching
        available_actual = [kw for kw in actual_keywords if kw not in used_actual]
        best_match, match_score = advanced_match_keywords(pred_kw, available_actual, model, similarity_threshold)
        
        if best_match and match_score >= similarity_threshold:
            matches.append((pred_kw, best_match, match_score))
            used_actual.add(best_match)
    
    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(pred_keywords) - true_positives
    false_negatives = len(actual_keywords) - true_positives
    
    precision = true_positives / len(pred_keywords) if pred_keywords else 0.0
    recall = true_positives / len(actual_keywords) if actual_keywords else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    if matches:
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
        "similarity_threshold": similarity_threshold,
        "matches": [(p, a, float(s)) for p, a, s in matches]  # Store all matches with scores
    }

def evaluate_with_embeddings(predicted_kws, actual_kws, model, threshold=0.75):
    # Note: This function is kept for backward compatibility
    # We now recommend using the more advanced evaluate_keywords function above
    matches = []
    used_actual = set()
    
    for pred_kw, pred_score in predicted_kws:
        best_match = None
        best_score = 0
        
        for actual_kw in actual_kws:
            if actual_kw in used_actual:
                continue
                
            # Use embeddings for similarity
            sim = embedding_similarity(pred_kw, actual_kw, model)
            
            if sim > best_score and sim >= threshold:
                best_score = sim
                best_match = actual_kw
        
        if best_match:
            matches.append((pred_kw, best_match, best_score))
            used_actual.add(best_match)
            
    return matches

def extract_hybrid_keywords(text, keybert_model, top_n=30):
    # Neural extraction with KeyBERT
    neural_keywords = keybert_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        use_mmr=True,
        diversity=0.6,
        top_n=top_n
    )
    
    # Filter out invalid keywords from neural extraction
    filtered_neural_keywords = []
    for kw, score in neural_keywords:
        if is_valid_keyword(kw):
            filtered_neural_keywords.append((kw, score))
        else:
            print(f"Filtered out invalid neural keyword: {kw}")
    
    if len(filtered_neural_keywords) < len(neural_keywords):
        print(f"Filtered out {len(neural_keywords) - len(filtered_neural_keywords)} invalid neural keywords")
    
    neural_keywords = filtered_neural_keywords
    
    # Statistical extraction with TF-IDF
    # For a single document, we'll treat paragraphs as "documents" for IDF calculation
    paragraphs = re.split(r'\n\s*\n', text)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words='english',
        min_df=1,  # Term must appear in at least 1 paragraph
        max_df=0.8  # Term appears in at most 80% of paragraphs
    )
    
    # Fit TF-IDF on paragraphs
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    
    # Combine TF-IDF scores across paragraphs to get document-level importance
    feature_names = vectorizer.get_feature_names_out()
    tfidf_sum = np.sum(tfidf_matrix, axis=0).A1
    
    # Get top TF-IDF terms
    top_indices = tfidf_sum.argsort()[-top_n:][::-1]
    tfidf_keywords = [(feature_names[idx], float(tfidf_sum[idx])) for idx in top_indices]
    
    # Combine keywords from both methods
    all_keywords = {}
    
    # Add neural keywords with normalization
    max_neural = max([score for _, score in neural_keywords]) if neural_keywords else 1.0
    for kw, score in neural_keywords:
        all_keywords[kw.lower()] = score / max_neural
    
    # Add TF-IDF keywords with normalization
    max_tfidf = max([score for _, score in tfidf_keywords]) if tfidf_keywords else 1.0
    for kw, score in tfidf_keywords:
        normalized_score = score / max_tfidf
        if kw.lower() in all_keywords:
            # If key exists in both, take weighted average
            all_keywords[kw.lower()] = all_keywords[kw.lower()] * 0.6 + normalized_score * 0.4
        else:
            all_keywords[kw.lower()] = normalized_score * 0.8  # Slightly lower weight for TF-IDF only terms
    
    # Convert back to list, sort, and take top_n
    final_keywords = [(kw, score) for kw, score in all_keywords.items()]
    final_keywords.sort(key=lambda x: x[1], reverse=True)
    
    return final_keywords[:top_n]

def main():
    print("Initializing KeyBERT with SentenceTransformer...")
    # Initialize keybert with a better sentence transformer model for educational content
    model_name = 'all-mpnet-base-v2'  # Better model for educational content
    print(f"Loading model: {model_name}")
    sentence_model = SentenceTransformer(model_name)
    keybert = KeyBERT(model=sentence_model)
    
    print(f"Processing {len(TRAIN_CHAPTERS)} training chapters...")
    # Load training chapters
    train_texts = []
    for chapter in TRAIN_CHAPTERS:
        text = load_chapter(chapter)
        if text:
            train_texts.append(text)
    
    # Extract keywords for each training chapter separately
    print("Extracting keywords from each training chapter...")
    train_keywords_by_chapter = {}
    for i, chapter in enumerate(TRAIN_CHAPTERS):
        text = train_texts[i]
        # Using lower diversity (0.5) to capture more semantically related terms
        train_keywords_by_chapter[chapter] = extract_hybrid_keywords(text, keybert, top_n=25)
    
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
    train_keywords = unique_train_keywords[:75]  # Increase number of keywords
    
    print("\nTop training keywords:")
    for kw, score in train_keywords[:10]:
        print(f"  - {kw}: {score:.4f}")
    
    # Evaluate on test chapters
    print(f"\nEvaluating on {len(TEST_CHAPTERS)} test chapters...")
    test_results = {}
    
    for chapter in TEST_CHAPTERS:
        test_text = load_chapter(chapter)
        if test_text:
            print(f"\nProcessing test chapter {chapter}...")
            # Extract keywords for test chapter with lower diversity
            test_keywords = extract_hybrid_keywords(test_text, keybert, top_n=25)
            
            # Evaluate against training keywords using hierarchical matching and embedding similarity
            metrics = evaluate_keywords(test_keywords, train_keywords, sentence_model, similarity_threshold=0.7)
            
            print(f"Results for chapter {chapter}:")
            for metric, value in metrics.items():
                if metric != "matches":  # Skip printing all matches
                    print(f"  - {metric}: {value:.4f}" if isinstance(value, float) else f"  - {metric}: {value}")
            
            # Print a few sample matches with their matching method
            print("\nSample matches (showing match technique):")
            for pred, actual, score in metrics["matches"][:5]:
                technique = "Exact match" if score >= 0.95 else \
                           "Containment" if score >= 0.8 else \
                           "Word overlap" if score >= 0.7 else \
                           "Embedding similarity"
                print(f"  - '{pred}' → '{actual}' ({technique}, score: {score:.3f})")
            
            test_results[f"chapter_{chapter}"] = {
                "keywords": [(kw, float(score)) for kw, score in test_keywords],  # Convert numpy floats to regular floats
                "metrics": {k: v for k, v in metrics.items() if k != "matches"},  # Don't store all matches in JSON
                "sample_matches": [(p, a, float(s)) for p, a, s in metrics["matches"][:10]]  # Store just a few examples
            }
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_name_short = model_name.split('/')[-1]
    output_file = RESULTS_DIR / f"keybert_{model_name_short}_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "train_keywords": [(kw, float(score)) for kw, score in train_keywords],
            "test_results": test_results,
            "parameters": {
                "model": model_name,
                "top_n": 25,
                "diversity": 0.5,
                "similarity_threshold": 0.7,
                "matching_approach": "hierarchical+embedding"
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 