#!/usr/bin/env python3
import os
import glob
import re
from pathlib import Path

def clean_soft_hyphens(text):
    """
    Clean text by removing soft hyphens that break words across lines.
    
    Replaces:
    - "‐ " (soft hyphen + space) with "" to rejoin broken words
    - any remaining "‐" (soft hyphen) with "" to normalize hyphens
    """
    # Count occurrences for reporting
    hyphen_space_count = text.count("‐ ")
    remaining_hyphen_count = text.count("‐") - hyphen_space_count
    
    # Replace soft hyphen followed by space (indicating word breaks)
    cleaned = text.replace("‐ ", "")
    
    # Replace any remaining soft hyphens
    cleaned = cleaned.replace("‐", "")  # Remove them completely
    # Alternative: replace with regular hyphens
    # cleaned = cleaned.replace("‐", "-")
    
    return cleaned, hyphen_space_count, remaining_hyphen_count

def process_file(filepath):
    """Process a single file to clean soft hyphens."""
    try:
        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        cleaned_content, hyphen_space_count, remaining_hyphen_count = clean_soft_hyphens(content)
        
        # Only write back if changes were made
        if hyphen_space_count > 0 or remaining_hyphen_count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            print(f"Fixed {filepath.name}:")
            print(f"  - Rejoined {hyphen_space_count} broken words")
            print(f"  - Removed {remaining_hyphen_count} other soft hyphens")
            return True
        else:
            print(f"No soft hyphens found in {filepath.name}")
            return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Find and process all chapter files in the textbook directory."""
    textbook_dir = Path("textbook")
    
    if not textbook_dir.exists() or not textbook_dir.is_dir():
        print(f"Error: Directory '{textbook_dir}' not found!")
        return
    
    # Find all chapter files (ch*.txt)
    chapter_files = list(textbook_dir.glob("ch*.txt"))
    chapter_files.sort()  # Sort files to process in order
    
    if not chapter_files:
        print(f"No chapter files found in {textbook_dir}")
        return
    
    print(f"Found {len(chapter_files)} chapter files. Starting cleanup...")
    
    # Process each file
    fixed_files = 0
    for filepath in chapter_files:
        if process_file(filepath):
            fixed_files += 1
    
    print(f"\nCompleted processing {len(chapter_files)} files.")
    print(f"Fixed soft hyphens in {fixed_files} files.")

if __name__ == "__main__":
    main() 