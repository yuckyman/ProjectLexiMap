# textbook keyword extraction evaluation

this project evaluates the performance of keybert for keyword extraction by comparing extracted keywords against a textbook's index. it's designed to assess how well automated keyword extraction matches human-curated keywords.

## project structure

```
.
├── README.md
├── requirements.txt
├── keybert_trainer.py       # initial implementation for training keybert
├── keybert_evaluator.py     # main evaluation script
├── textbook/               
│   ├── ch1.txt - ch19.txt  # textbook chapters
│   ├── index.txt           # complete textbook index
│   └── index_by_chapter.txt # index organized by chapter
```

## setup

1. create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. install dependencies:
```bash
pip install -r requirements.txt
```

## implementation details

### keyword extraction
- uses keybert with sentence-transformers (`all-MiniLM-L6-v2` model)
- extracts up to 50 keywords per chapter
- uses n-grams (1-3 words) to capture phrases
- implements diversity in keyword selection (diversity=0.7)

### evaluation metrics
- precision: ratio of correctly identified keywords to total predicted keywords
- recall: ratio of correctly identified keywords to total actual keywords
- f1 score: harmonic mean of precision and recall

### keyword matching
the evaluation implements fuzzy matching to handle variations in keyword representation:

1. normalization:
   - converts to lowercase
   - removes special characters
   - standardizes whitespace

2. similarity scoring:
   - exact matches after normalization (score: 1.0)
   - contained terms (score: 0.9)
   - fuzzy string matching using sequence matcher

3. matching criteria:
   - similarity threshold: 0.8
   - prevents duplicate matches
   - handles partial matches and variations

## usage

run the evaluation:
```bash
python3 keybert_evaluator.py
```

this will:
1. load the textbook chapters and index
2. extract keywords from each chapter
3. compare against ground truth (index)
4. generate evaluation metrics
5. create visualizations
6. save detailed results to `keybert_evaluation_results.json`

## output

the script produces:
1. console output with:
   - per-chapter metrics
   - matched keywords with similarity scores
   - average metrics across all chapters

2. visualization:
   - bar plot comparing precision, recall, and f1 scores across chapters
   - saved as `keybert_evaluation.png`

3. detailed results in json format:
   - extracted keywords with confidence scores
   - ground truth keywords
   - matching results
   - evaluation metrics

## training vs testing split

the evaluation is designed to work with specific chapter splits:

- training chapters: 1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19
- test chapters: 6, 10, 11, 12

this split allows for evaluating the model's performance on both seen and unseen content.

## future improvements

1. semantic similarity:
   - incorporate word embeddings for similarity matching
   - use domain-specific embeddings

2. keyword extraction:
   - experiment with different keybert parameters
   - try other extraction algorithms (rake, yake, etc.)
   - combine multiple approaches

3. evaluation:
   - add more granular metrics
   - implement cross-validation
   - analyze performance by keyword type/length 
