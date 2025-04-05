# textbook keyword extraction evaluation

this project evaluates the performance of keybert for keyword extraction by comparing extracted keywords against a textbook's index. it's designed to assess how well automated keyword extraction matches human-curated keywords.

## project structure

```
.
├── README.md
├── requirements.txt
├── keybert_trainer.py       # implementation for training keybert with enhanced methods
├── keybert_evaluator.py     # main evaluation script
├── mindmap_generator.py     # creates interactive knowledge graph visualizations
├── index_mindmap_generator.py # creates ground truth mindmaps from the index
├── textbook/               
│   ├── ch1.txt - ch19.txt  # textbook chapters
│   ├── index.txt           # complete textbook index
│   └── index_by_chapter.txt # index organized by chapter
├── results/                # evaluation results directory
│   ├── mindmaps/          # interactive html mindmaps
│   └── *.png, *.json      # final results and visualizations
└── cache/                  # cached data for faster processing
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
python keybert_trainer.py

# Evaluate extraction quality against textbook index
python keybert_evaluator.py
```

configurable parameters in both scripts:
- `top_n`: number of keywords to extract (default now 100)
- `diversity`: diversity factor for keyword selection (default now 0.7)
- `similarity_threshold`: threshold for keyword matching (default 0.7)

the scripts will:
- process all training chapters (1-5, 7-9, 13-19)
- extract keywords using the default model (all-mpnet-base-v2)
- combine neural and statistical approaches for better extraction
- evaluate against test chapters (6, 10, 11, 12)
- save results to the results directory
- cache processed data and embeddings for faster subsequent runs

### generating knowledge graph visualizations

after extracting keywords, you can create interactive mindmaps that visualize the relationships between keywords:

```bash
# Generate mindmap from extracted keywords
python mindmap_generator.py --chapters 6 10 11 12 --results results/keybert_all-mpnet-base-v2_results_TIMESTAMP.json --output results/mindmaps/keybert_mindmap.html

# Generate ground truth mindmap from index
python index_mindmap_generator.py --chapters 6 10 11 12 --output results/mindmaps/ground_truth_mindmap.html
```

options:
- `--chapters`: chapter numbers to include in the mindmap (e.g., `--chapters 6 10 11 12`)
- `--results`: path to keybert results file (generated from training)
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
  - `all-mpnet-base-v2` (default)
- extracts up to 100 keywords per chapter (configurable)
- uses n-grams (1-3 words) to capture phrases
- implements high diversity in keyword selection (0.7)
- **hybrid approach**: combines neural (transformer) and statistical (TF-IDF) methods
- **paragraph-aware processing**: treats paragraphs as separate documents for TF-IDF calculation
- **embedding caching**: caches keyword embeddings to improve performance

### evaluation metrics
- precision: ratio of correctly identified keywords to total predicted keywords
- recall: ratio of correctly identified keywords to total actual keywords
- f1 score: harmonic mean of precision and recall
- typical performance: ~13% precision, ~10% recall (index terms are chosen differently)

### keyword matching
the evaluation implements advanced methods for matching keywords:

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
   - configurable similarity threshold (default: 0.7)

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

1. **increased extraction volume**:
   - extracts 100 keywords per chapter (up from 30-50)
   - keeps 300 unique training keywords (up from 75-150)
   - better coverage of relevant terms

2. **enhanced diversity**:
   - diversity parameter increased to 0.7
   - captures a wider range of concepts
   - reduces redundancy in extracted terms

3. **relaxed similarity matching**:
   - reduced similarity threshold to 0.7
   - better identifies related concepts
   - improves recall without sacrificing precision

4. **chapter-based visualization**:
   - organizes mindmaps by chapter instead of clusters
   - provides clearer structure and context
   - makes comparison between extracted and index terms easier

5. **ground truth visualization**:
   - added index-based mindmap generation
   - provides baseline for comparison
   - helps assess extraction quality visually 