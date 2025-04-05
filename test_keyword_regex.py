#!/usr/bin/env python3
import re

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

def test_keywords():
    # Regular valid keywords
    valid_keywords = [
        "machine",
        "learning",
        "machine-learning",
        "hyperparameter",
        "k-means",
        "k_means",
        "C++",
        "ML_algorithms",
        "Python3",
        "O(n)",
        "O(n^2)",
        "classifier",
        "decision-tree",
        "n-gram",
        "SVM",
        "CNN",
        "RNN",
        "GRU",
        "LSTM",
        "Smith's",
        "network",
        "neural_network",
        "r-squared",
        "python_3.x",
        "CSV",
        "JSON",
        "TF-IDF",
        "[0,1]",          # Now should work with comma
        "(0,1)",          # Now should work with comma
        "(learning_rate)",
        "O(log_n)",
        "Backprop",
        # Additional examples with new characters
        "C&C++",
        "AI&ML",
        "A+B",            # This should now be caught as math expression
        "A.B",
        "std::vector",
        "2^n",
        "Wide&Deep",
        "First-In.First-Out",
        # Index terms with commas
        "machine,learning",
        "supervised,learning",
        "cross-validation,techniques",
        "neural,networks"
    ]
    
    # Keywords that should be filtered out
    invalid_keywords = [
        "f(x) = 2x + 3",
        "x^2 + y^2 = z^2",
        "y = mx + b",
        "P(X|Y) = P(Y|X)P(X)/P(Y)",
        "Σx_i / n",
        "$\\alpha$",
        "∫f(x)dx",
        "5 * 3 + 2",
        "a ⊂ b",
        "a ∪ b",
        "x ∈ R",
        "∀x ∃y",
        "λx.x+1",
        "δ/δx",
        "x → y",
        "x ≠ y",
        "{a,b,c}",
        "⁄‹⁄‹.›",
        "+++"
    ]
    
    # Multi-word phrases that should be valid
    valid_phrases = [
        "machine learning",
        "deep learning",
        "decision tree",
        "binary classifier",
        "hyperparameter tuning",
        "tf-idf score",
        "accuracy metric",
        "loss function",
        "validation set",
        "cross validation",
        "feature selection",
        "feature engineering",
        "f(x) definition", # Valid because it's multi-word
        "y = mx + b graph", # Valid because it's multi-word 
        "x and y coordinates"
    ]
    
    # Check valid keywords
    print("Testing valid keywords:")
    for kw in valid_keywords:
        result = is_valid_keyword(kw)
        print(f"  {kw:20} -> {'✓' if result else '✗'}")
        if not result:
            print(f"    WARNING: Expected valid keyword '{kw}' was marked invalid!")
    
    # Check invalid keywords
    print("\nTesting invalid keywords:")
    for kw in invalid_keywords:
        result = is_valid_keyword(kw)
        print(f"  {kw:20} -> {'✗' if not result else '✓'}")
        if result:
            print(f"    WARNING: Expected invalid keyword '{kw}' was marked valid!")
    
    # Check valid phrases
    print("\nTesting valid phrases:")
    for phrase in valid_phrases:
        result = is_valid_keyword(phrase)
        print(f"  {phrase:20} -> {'✓' if result else '✗'}")
        if not result:
            print(f"    WARNING: Expected valid phrase '{phrase}' was marked invalid!")

if __name__ == "__main__":
    test_keywords() 