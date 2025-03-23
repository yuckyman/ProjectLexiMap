# textbook keyword extraction evaluation

this project evaluates the performance of keybert for keyword extraction by comparing extracted keywords against a textbook's index. it's designed to assess how well automated keyword extraction matches human-curated keywords.

## project structure

```
.
├── README.md
├── requirements.txt
├── keybert_trainer.py       # implementation for training keybert with enhanced methods
├── keybert_evaluator.py     # main evaluation script
├── textbook/               
│   ├── ch1.txt - ch19.txt  # textbook chapters
│   ├── index.txt           # complete textbook index
│   └── index_by_chapter.txt # index organized by chapter
├── results/                # evaluation results directory
│   ├── interim/           # interim results during evaluation
│   └── *.png, *.json      # final results and visualizations
└── cache/                  # cached data for faster processing
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

## running the scripts

### unified script (recommended)
we provide a unified script that combines both training and evaluation functionality:

```bash
python3 keybert_unified.py [command] [options]
```

available commands:
- `train`: train the model on training chapters
- `evaluate`: evaluate model performance against ground truth keywords

#### training with the unified script
```bash
python3 keybert_unified.py train --model all-mpnet-base-v2 --top_n 30 --diversity 0.5
```

options:
- `--model`: sentence transformer model to use (default: all-mpnet-base-v2)
- `--top_n`: number of keywords to extract per chapter (default: 50)
- `--diversity`: diversity factor for keyword selection (0-1) (default: 0.5)
- `--similarity_threshold`: threshold for keyword matching (default: 0.8)
- `--timeout`: timeout in seconds for keyword extraction (default: 120)
- `--clear_cache`: clear cache before running
- `--skip_evaluation`: skip evaluation on test chapters
- `--clean_results`: remove old training results before running

#### evaluation with the unified script
```bash
python3 keybert_unified.py evaluate --model all-mpnet-base-v2 --top_n 30 --diversity 0.7
```

additional evaluation options:
- `--chapters`: specific chapters to evaluate (e.g., `--chapters 1 2 3`)
- `--no_interim`: disable saving interim results
- `--clean_results`: remove old evaluation results before running

### legacy scripts
alternatively, you can use the original separate scripts:

#### training
to train the model on the training chapters (1-5, 7-9, 13-19):
```bash
python3 keybert_trainer.py
```

this will:
- process all training chapters
- extract keywords using the default model (all-mpnet-base-v2)
- combine neural and statistical approaches for better extraction
- save results to the results directory
- cache processed data and embeddings for faster subsequent runs

#### evaluation
basic usage with default models:
```bash
python3 keybert_evaluator.py
```

evaluate specific models:
```bash
python3 keybert_evaluator.py --model all-mpnet-base-v2 all-MiniLM-L6-v2
```

advanced usage with options:
```bash
python3 keybert_evaluator.py \
  --model all-mpnet-base-v2 all-MiniLM-L6-v2 \
  --top_n 30 \
  --diversity 0.7 \
  --similarity_threshold 0.8 \
  --timeout 120 \
  --clean_results
```

legacy command-line options:
- `--model`: list of sentence transformer models to evaluate
- `--top_n`: number of keywords to extract (default: 50)
- `--diversity`: diversity factor for keyword selection (0-1, default: 0.5)
- `--similarity_threshold`: threshold for keyword matching (default: 0.8)
- `--timeout`: timeout in seconds for keyword extraction (default: 120)
- `--clear_cache`: clear cached data before running
- `--clean_results`: remove old results before running
- `--no_interim`: disable saving interim results

## implementation details

### keyword extraction
- uses keybert with multiple sentence-transformer models:
  - `all-mpnet-base-v2` (default)
  - `all-MiniLM-L6-v2`
  - `paraphrase-multilingual-MiniLM-L12-v2`
  - support for custom models (including domain-specific models)
- extracts up to 50 keywords per chapter (configurable)
- uses n-grams (1-3 words) to capture phrases
- implements diversity in keyword selection (configurable, default=0.5)
- **hybrid approach**: combines neural (transformer) and statistical (TF-IDF) methods
- **paragraph-aware processing**: treats paragraphs as separate documents for TF-IDF calculation
- **embedding caching**: caches keyword embeddings to improve performance

### evaluation metrics
- precision: ratio of correctly identified keywords to total predicted keywords
- recall: ratio of correctly identified keywords to total actual keywords
- f1 score: harmonic mean of precision and recall
- comparative analysis across different models

### keyword matching
the evaluation implements two methods for matching keywords:

1. **string-based matching**:
   - normalization (lowercase, remove special characters)
   - exact matches after normalization (score: 1.0)
   - contained terms (score: 0.9)
   - fuzzy string matching using sequence matcher

2. **embedding-based similarity**:
   - uses transformer model to create vector representations
   - calculates cosine similarity between keyword vectors
   - captures semantic relationships between terms
   - better handles synonyms and related concepts
   - configurable similarity threshold (default: 0.75)

3. matching criteria:
   - similarity threshold: configurable
   - prevents duplicate matches
   - handles partial and semantic matches

## output

the script produces for each model:

1. console output with:
   - per-chapter metrics
   - matched keywords with similarity scores
   - average metrics across all chapters
   - execution time and progress updates

2. visualizations:
   - individual model performance plots
   - comparative plot across all evaluated models
   - saved in the `results` directory with timestamps

3. detailed results:
   - individual model results: `keybert_[model]_results_[timestamp].json`
   - individual plots: `keybert_[model]_evaluation_[timestamp].png`
   - comparison plot: `keybert_model_comparison_[timestamp].png`
   - interim results (optional): saved during evaluation

4. cached data:
   - extracted keywords
   - processed chapters
   - keyword embeddings
   - stored in `cache` directory for faster subsequent runs

## training vs testing split

the evaluation is designed to work with specific chapter splits:

- training chapters: 1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19
- test chapters: 6, 10, 11, 12

this split allows for evaluating the model's performance on both seen and unseen content.

## implemented improvements

1. **embedding-based similarity**:
   - uses transformer model embeddings instead of string similarity
   - captures semantic relationships between terms
   - caches embeddings for better performance

2. **hybrid keyword extraction**:
   - combines transformer models with TF-IDF
   - neural methods capture semantic meaning
   - statistical methods identify domain-specific terminology
   - weighted combination for better results

3. **domain-specific preprocessing**:
   - enhanced stopword lists for educational content
   - paragraph-aware text processing
   - normalization of technical terms

## future improvements

1. **model selection**:
   - try domain-specific models like scibert
   - experiment with fine-tuning on educational content
   - benchmark different model architectures

2. **contextual understanding**:
   - implement hierarchical extraction (sentence → paragraph → chapter)
   - better handling of sections and subsections

3. **evaluation**:
   - implement precision@k metrics
   - add cross-validation
   - analyze performance by keyword type/length

4. **hyperparameter optimization**:
   - systematic grid search for optimal parameters
   - automatic parameter selection based on content 