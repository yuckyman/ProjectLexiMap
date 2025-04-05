#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path

def clean_soft_hyphens(text):
    """
    Clean text by removing soft hyphens that break words across lines.
    Replaces:
    - "‐ " (soft hyphen + space) with "" to rejoin broken words
    - any remaining "‐" (soft hyphen) with "" to normalize hyphens
    """
    # Replace soft hyphen followed by space (indicating word breaks)
    cleaned = text.replace("‐ ", "")
    
    # Replace any remaining soft hyphens
    cleaned = cleaned.replace("‐", "")  # Remove them completely
    
    return cleaned

def clean_keyword(keyword):
    """Clean a single keyword by removing soft hyphens."""
    return clean_soft_hyphens(keyword)

def fix_cache_keywords():
    """
    Fix any cached keyword files to remove soft hyphens.
    This ensures that the extraction won't need to run again.
    """
    cache_dir = Path("cache")
    
    if not cache_dir.exists() or not cache_dir.is_dir():
        print(f"Cache directory ({cache_dir}) not found!")
        return
    
    # Find all keyword cache files
    keyword_files = list(cache_dir.glob("keywords_ch*.json"))
    
    if not keyword_files:
        print("No keyword cache files found.")
        return
    
    print(f"Found {len(keyword_files)} keyword cache files. Cleaning keywords...")
    
    fixed_files = 0
    for file_path in keyword_files:
        try:
            # Read the cached keywords
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords_data = json.load(f)
            
            # Check if it's a list of lists/tuples format or another format
            if isinstance(keywords_data, list):
                has_changes = False
                cleaned_keywords = []
                
                for item in keywords_data:
                    if isinstance(item, list) and len(item) == 2:
                        keyword, score = item
                        cleaned_keyword = clean_keyword(keyword)
                        if cleaned_keyword != keyword:
                            has_changes = True
                        cleaned_keywords.append([cleaned_keyword, score])
                    else:
                        # Keep as is if structure is unexpected
                        cleaned_keywords.append(item)
                
                if has_changes:
                    # Write the cleaned data back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_keywords, f, indent=2)
                    print(f"Fixed keywords in {file_path.name}")
                    fixed_files += 1
                else:
                    print(f"No changes needed in {file_path.name}")
            else:
                print(f"Unexpected format in {file_path.name}, skipping")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Completed cleaning {fixed_files} keyword cache files.")

def fix_index_keywords():
    """
    Fix the cached index keywords file if it exists.
    This ensures that the ground truth keywords are also cleaned.
    """
    cache_dir = Path("cache")
    index_cache = cache_dir / "index_keywords.json"
    
    if not index_cache.exists():
        print("No cached index keywords found.")
        return
    
    try:
        print("Cleaning cached index keywords...")
        with open(index_cache, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        has_changes = False
        # Process each chapter's keywords
        for chapter, keywords in index_data.items():
            cleaned_keywords = []
            for keyword in keywords:
                cleaned_keyword = clean_keyword(keyword)
                if cleaned_keyword != keyword:
                    has_changes = True
                cleaned_keywords.append(cleaned_keyword)
            
            if has_changes:
                index_data[chapter] = cleaned_keywords
        
        if has_changes:
            # Write the cleaned data back
            with open(index_cache, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            print("Fixed index keywords cache.")
        else:
            print("No changes needed in index keywords.")
            
    except Exception as e:
        print(f"Error processing index keywords: {e}")

def fix_evaluation_results():
    """
    Fix any evaluation result files to clean keywords in results.
    """
    results_dir = Path("results")
    
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Results directory ({results_dir}) not found!")
        return
    
    # Find all result JSON files
    result_files = list(results_dir.glob("keybert_*_results_*.json"))
    
    if not result_files:
        print("No evaluation result files found.")
        return
    
    print(f"Found {len(result_files)} result files. Cleaning keywords...")
    
    fixed_files = 0
    for file_path in result_files:
        try:
            # Read the result file
            with open(file_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            has_changes = False
            
            # Fix train keywords if present
            if "train_keywords" in results_data:
                train_keywords = results_data["train_keywords"]
                if isinstance(train_keywords, list):
                    for i, item in enumerate(train_keywords):
                        if isinstance(item, list) and len(item) == 2:
                            keyword, score = item
                            cleaned_keyword = clean_keyword(keyword)
                            if cleaned_keyword != keyword:
                                train_keywords[i][0] = cleaned_keyword
                                has_changes = True
            
            # Fix chapter results if present
            if "test_results" in results_data:
                for chapter, chapter_data in results_data["test_results"].items():
                    if "keywords" in chapter_data:
                        keywords = chapter_data["keywords"]
                        for i, item in enumerate(keywords):
                            if isinstance(item, list) and len(item) == 2:
                                keyword, score = item
                                cleaned_keyword = clean_keyword(keyword)
                                if cleaned_keyword != keyword:
                                    keywords[i][0] = cleaned_keyword
                                    has_changes = True
                    
                    if "sample_matches" in chapter_data:
                        matches = chapter_data["sample_matches"]
                        for i, match in enumerate(matches):
                            if isinstance(match, list) and len(match) >= 2:
                                pred, actual = match[0], match[1]
                                cleaned_pred = clean_keyword(pred)
                                cleaned_actual = clean_keyword(actual)
                                if cleaned_pred != pred or cleaned_actual != actual:
                                    matches[i][0] = cleaned_pred
                                    matches[i][1] = cleaned_actual
                                    has_changes = True
            
            # Fix chapter results if in other format
            if "chapter_metrics" in results_data:
                for chapter, chapter_data in results_data.get("chapter_metrics", {}).items():
                    if "extracted_keywords" in chapter_data:
                        extracted = chapter_data["extracted_keywords"]
                        if isinstance(extracted, dict):
                            # Dictionary format (keyword -> score)
                            cleaned_extracted = {}
                            for keyword, score in extracted.items():
                                cleaned_keyword = clean_keyword(keyword)
                                if cleaned_keyword != keyword:
                                    has_changes = True
                                    cleaned_extracted[cleaned_keyword] = score
                                else:
                                    cleaned_extracted[keyword] = score
                            chapter_data["extracted_keywords"] = cleaned_extracted
                        elif isinstance(extracted, list):
                            # List format (could be list of lists or tuples)
                            for i, item in enumerate(extracted):
                                if isinstance(item, (list, tuple)) and len(item) == 2:
                                    keyword, score = item
                                    cleaned_keyword = clean_keyword(keyword)
                                    if cleaned_keyword != keyword:
                                        extracted[i] = [cleaned_keyword, score]
                                        has_changes = True
            
            if has_changes:
                # Write the cleaned data back
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2)
                print(f"Fixed keywords in {file_path.name}")
                fixed_files += 1
            else:
                print(f"No changes needed in {file_path.name}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Completed cleaning {fixed_files} result files.")

def update_normalize_keyword_function():
    """
    Inform user about updating the normalize_keyword function in their code.
    """
    print("\nIMPORTANT: You should also update your normalize_keyword function in")
    print("keybert_evaluator.py and keybert_trainer.py to handle soft hyphens.")
    print("\nAdd this code to the normalize_keyword function in both files:")
    print("""
def normalize_keyword(keyword: str) -> str:
    # First clean any soft hyphens before normalization
    keyword = keyword.replace("‐ ", "").replace("‐", "")
    
    # Then perform the usual normalization
    keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    return keyword
    """)

def main():
    print("Starting to fix keyword files and caches...")
    
    # Fix cached keywords from extraction
    fix_cache_keywords()
    
    # Fix index keywords (ground truth)
    fix_index_keywords()
    
    # Fix evaluation results
    fix_evaluation_results()
    
    # Inform about code updates
    update_normalize_keyword_function()
    
    print("\nDONE! All keyword files have been cleaned.")
    print("For best results:")
    print("1. Run 'python fix_hyphens.py' to clean all chapter text files")
    print("2. Update the normalize_keyword function as shown above")
    print("3. Re-run your evaluation to see improved matching metrics")

if __name__ == "__main__":
    main() 