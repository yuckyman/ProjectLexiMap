# Data preparation for fine-tuning sentence-transformer on textbook content
import os
import re
import random
import nltk
from typing import Dict, List
from sentence_transformers import InputExample
from finetune.config import TEXTBOOK_DIR, INDEX_FILE, TRAIN_CHAPTERS

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize


def load_index_keywords() -> Dict[int, List[str]]:
    """Load ground truth keywords per chapter from the index file."""
    keywords: Dict[int, List[str]] = {}
    current_chap = None
    try:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('chapter '):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        current_chap = int(parts[1])
                        keywords[current_chap] = []
                else:
                    if current_chap is not None and line:
                        keywords[current_chap].append(line)
    except Exception as e:
        raise RuntimeError(f"Failed to load index keywords: {e}")
    return keywords


def load_chapter_text(chapter_num: int) -> str:
    """Load the raw text for a given chapter number."""
    path = os.path.join(TEXTBOOK_DIR, f'ch{chapter_num}.txt')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load chapter {chapter_num}: {e}")


def get_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on blank lines."""
    paras = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paras if p.strip()]


def generate_examples() -> List[InputExample]:
    """
    Generate triplet examples (context, positive keyword, negative keyword) for fine-tuning.
    Anchor = paragraph text, Positive = true keyword in that paragraph,
    Negative = a random keyword from a different chapter.
    """
    index_data = load_index_keywords()
    examples: List[InputExample] = []

    for chap in TRAIN_CHAPTERS:
        text = load_chapter_text(chap)
        paras = get_paragraphs(text)
        keywords = index_data.get(chap, [])
        if not keywords:
            continue
        for para in paras:
            # find positive keywords that occur in this paragraph (case-insensitive match)
            para_lower = para.lower()
            positives = [kw for kw in keywords if kw.lower() in para_lower]
            for pos in positives:
                # sample a negative from a different chapter
                other_chaps = [c for c in TRAIN_CHAPTERS if c != chap and index_data.get(c)]
                if not other_chaps:
                    continue
                neg_chap = random.choice(other_chaps)
                neg_kw = random.choice(index_data[neg_chap])
                examples.append(InputExample(texts=[para, pos, neg_kw]))

    random.shuffle(examples)
    return examples 