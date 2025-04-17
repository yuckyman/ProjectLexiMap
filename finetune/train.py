#!/usr/bin/env python3
"""
Fine-tune a SentenceTransformer model on textbook content using triplet loss.
"""
from sentence_transformers import SentenceTransformer, losses, models
from torch.utils.data import DataLoader
from finetune.config import (
    BASE_MODEL_NAME,
    MODEL_OUT_DIR,
    BATCH_SIZE,
    EPOCHS,
    WARMUP_STEPS,
    LEARNING_RATE,
)
from finetune.data_prep import generate_examples


def main():
    # Load base model
    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)

    # Prepare training examples
    print("Generating fine-tuning examples...")
    train_examples = generate_examples()
    print(f"Total examples: {len(train_examples)}")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.TripletLoss(model=model)

    # Fine-tune
    print(f"Starting training for {EPOCHS} epochs, batch size {BATCH_SIZE}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=MODEL_OUT_DIR,
        show_progress_bar=True,
    )

    print(f"Model fine-tuned and saved to: {MODEL_OUT_DIR}")


if __name__ == '__main__':
    main() 