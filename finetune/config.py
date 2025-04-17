# Configuration for fine-tuning sentence-transformer on textbook content
import os

# Base model to fine-tune (will be loaded by SentenceTransformer)
BASE_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

# Directory paths
TEXTBOOK_DIR = 'textbook'
INDEX_FILE = os.path.join(TEXTBOOK_DIR, 'index_by_chapter.txt')

# Chapters to use for fine-tuning
TRAIN_CHAPTERS = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]

# Output directory for the fine-tuned model
MODEL_OUT_DIR = os.path.join('models', 'mpnet_textbook_tuned')

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 3
WARMUP_STEPS = 100
LEARNING_RATE = 2e-5

# Ensure output dir exists
os.makedirs(MODEL_OUT_DIR, exist_ok=True) 