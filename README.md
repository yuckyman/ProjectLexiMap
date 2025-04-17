# textbook keyword extraction evaluation

this project evaluates the performance of keybert for keyword extraction by comparing extracted keywords against a textbook's index. it's designed to assess how well automated keyword extraction matches human-curated keywords.

## project structure

```
.
├── README.md
├── requirements.txt
├── jupyter_trainer.py       # primary implementation for training/evaluation
├── keybert_trainer.py       # deprecated implementation (kept for reference)
├── finetune/              # code for fine-tuning the sentence transformer model
│   ├── config.py          # fine-tuning configuration settings
│   ├── data_prep.py       # data preparation for fine-tuning
│   └── train.py          # main fine-tuning script
├── mindmap_generator.py     # creates interactive knowledge graph visualizations
├── index_mindmap_generator.py # creates ground truth mindmaps from the index
├── textbook/               
│   ├── ch1.txt - ch19.txt  # textbook chapters
│   ├── index.txt           # complete textbook index
│   └── index_by_chapter.txt # index organized by chapter
├── results/                # evaluation results directory
│   ├── mindmaps/          # interactive html mindmaps
│   └── *.png, *.json      # final results and visualizations
├── cache/                  # cached data for faster processing
├── models/                # directory for saving fine-tuned models
└── checkpoints/           # interim directory during training
```

## setup

1. create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. install dependencies:
```bash
pip install -r requirements.txt
```

## running the scripts

### keyword extraction and evaluation

```bash
# Extract keywords with optimized parameters
python jupyter_trainer.py
```

configurable parameters in `jupyter_trainer.py`:
-  `top_n`: number of keywords to extract (default: 75)
-  `diversity`: diversity factor for keyword selection (default: 0.6)
-  `EVAL_SIMILARITY_THRESHOLD`: threshold for keyword matching (default: 0.75)

the script will:
-  process all training chapters (1-5, 7-9, 13-19)
-  extract keywords using the fine-tuned model (mpnet_textbook_tuned)
-  combine neural and statistical approaches for better extraction
-  evaluate against test chapters (6, 10, 11, 12)
-  save results to the results directory
-  cache processed data and embeddings for faster subsequent runs
-  generate a metrics plot alongside the JSON results

> **Note:** Parameter values may vary from those documented as we continue to optimize results. The implementation in `jupyter_trainer.py` represents our current best approach, while `keybert_trainer.py` is kept for reference only.

### fine-tuning sentence transformers

to further improve keyword extraction, you can fine-tune the underlying sentence transformer model on your textbook content:

```bash
# Fine-tune the sentence transformer model using triplet loss
python -m finetune.train

# After fine-tuning, the model will be available at:
# models/mpnet_textbook_tuned
```

the fine-tuning process:
1. processes all training chapters (1-5, 7-9, 13-19).
2. locates ground truth keywords in paragraph contexts.
3. creates triplet examples (context, positive keyword, negative keyword).
4. trains the model using triplet loss to learn domain-specific relationships.

configurable parameters (in `finetune/config.py`):
- `BASE_MODEL_NAME`: base model to fine-tune (default: 'sentence-transformers/all-mpnet-base-v2')
- `BATCH_SIZE`: batch size for training (default: 16)
- `EPOCHS`: number of training epochs (default: 3)
- `LEARNING_RATE`: learning rate for fine-tuning (default: 2e-5)

requirements:
- requires pytorch and the `accelerate` library
- gpu recommended for faster training

### generating knowledge graph visualizations

after extracting keywords, you can create interactive mindmaps that visualize the relationships between keywords:

```bash
# Generate mindmap from extracted keywords
python mindmap_generator.py --chapters 6 10 11 12 --results results/evaluation_TIMESTAMP.json --output results/mindmaps/keybert_mindmap.html

# Generate ground truth mindmap from index
python index_mindmap_generator.py --chapters 6 10 11 12 --output results/mindmaps/ground_truth_mindmap.html
```

options:
- `--chapters`: chapter numbers to include in the mindmap (e.g., `--chapters 6 10 11 12`)
- `--results`: path to evaluation results file (generated from jupyter_trainer.py)
- `--model`: sentence transformer model to use (default: all-mpnet-base-v2)
- `--similarity`: threshold for connecting keywords (0-1) (default: 0.65)
- `--max_keywords`: maximum number of keywords to include (default: 150)
- `--min_edge_weight`: minimum weight for edges (default: 0.6)
- `--output`: output html file path
- `--title`: title for the visualization
- `--no_cache`: disable embedding cache
- `--quiet`: suppress verbose output

how it works:
1. loads extracted keywords from evaluation results
2. creates embeddings for each keyword using sentence transformers
3. builds a knowledge graph where:
   - nodes = keywords
   - edges = semantic relationships (based on cosine similarity)
   - chapters = hub nodes with distinct colors
4. organizes keywords by chapter (represented as diamond nodes)
5. exports an interactive html visualization using pyvis

the visualization allows:
- interactive exploration of keyword relationships
- zooming and panning
- color-coded keywords by chapter
- hover information with details
- physics-based layout where related keywords naturally group together

## implementation details

### keyword extraction
- uses keybert with sentence-transformer model:
  - fine-tuned `mpnet_textbook_tuned` model (defaults to `all-mpnet-base-v2` if not available)
- extracts keywords per chapter (75-100 configurable)
- uses n-grams (1-3 words) to capture phrases
- implements diversity in keyword selection (0.6-0.7)
- filters out very short keywords (≤2 characters)
- **sentence chunking**: splits text into sentence-based chunks using regex-based splitter
- **hybrid approach**: combines neural (transformer) and statistical (TF-IDF) methods:
  - Filters candidates using `is_valid_keyword` before merging
  - Combines scores with customizable weighting
- **paragraph-aware processing**: treats paragraphs as separate documents for TF-IDF calculation
- **embedding caching**: caches keyword embeddings to improve performance

### evaluation metrics
- precision: ratio of correctly identified keywords to total predicted keywords
- recall: ratio of correctly identified keywords to total actual keywords
- f1 score: harmonic mean of precision and recall
- **ground truth**: evaluates test chapter keywords against actual index keywords for that chapter (loaded from `index_by_chapter.txt`)
- similarity threshold: 0.75 (optimized through experimentation)

### keyword matching
the evaluation implements advanced methods for matching keywords:

1. **embedding-based similarity**:
   - uses transformer model to create vector representations
   - calculates cosine similarity between keyword vectors
   - captures semantic relationships between terms
   - better handles synonyms and related concepts
   - configurable similarity threshold (default: 0.75)

2. **string-based matching**:
   - normalization (lowercase, remove special characters)
   - exact matches after normalization (score: 1.0)
   - contained terms (score: 0.9)
   - fuzzy string matching using sequence matcher

3. **hierarchical matching**:
   - specialized matching for hierarchical terms in indices
   - handles variations in term order and phrasing
   - combines string and semantic methods

### visualization features

1. **chapter-based organization**:
   - keywords organized around chapter hub nodes
   - each chapter has a distinct color
   - chapter nodes shown as diamonds with larger font
   - keywords connected to their chapter with dashed lines

2. **index-based ground truth**:
   - `index_mindmap_generator.py` creates ground truth mindmaps
   - visualizes index terms organized by chapter
   - allows direct comparison with extracted keywords

3. **semantic connections**:
   - keywords connected based on semantic similarity
   - similar concepts cluster together visually
   - edge thickness indicates strength of relationship

## training vs testing split

the evaluation is designed to work with specific chapter splits:

- training chapters: 1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19
- test chapters: 6, 10, 11, 12

this split allows for evaluating the model's performance on both seen and unseen content.

## implemented improvements

1. **optimized extraction volume**:
   - extracts 75-100 keywords per chapter (tuned for precision/recall balance)
   - keeps up to 300 unique training keywords 
   - better coverage of relevant terms

2. **enhanced diversity**:
   - diversity parameter tuned between 0.6-0.7
   - captures a wider range of concepts
   - reduces redundancy in extracted terms

3. **length filtering**:
   - filters out very short keywords (≤2 characters)
   - improves precision by removing common short words
   - reduces false positives

4. **optimized similarity matching**:
   - similarity threshold tuned to 0.75
   - balances precision and recall
   - better identifies related concepts

5. **chapter-based visualization**:
   - organizes mindmaps by chapter instead of clusters
   - provides clearer structure and context
   - makes comparison between extracted and index terms easier

6. **ground truth visualization**:
   - added index-based mindmap generation
   - provides baseline for comparison
   - helps assess extraction quality visually

7. **improved extraction pipeline:**
   - switched to sentence-based chunking for better semantic coherence
   - tuned hybrid extraction to filter noise and prioritize neural results
   - enhanced stopword list to remove common non-keywords

8. **model fine-tuning:**
   - implemented capability to fine-tune the sentence transformer model on textbook content
   - uses triplet loss to learn domain-specific relationships between context and keywords
   - created a custom model that better understands the textbook's terminology and style 