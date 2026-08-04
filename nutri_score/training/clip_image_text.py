import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from nutri_score.image_utils import image_from_base64
from nutri_score.paths import DATASET_DIR, MODEL_DIR

MODEL_NAME = "openai/clip-vit-base-patch32"


class ImageTextDataset(Dataset):
    def __init__(self, dataframe, processor, image_column="filename", text_column="post_mood_PNN"):
        self.df = dataframe
        self.processor = processor
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = image_from_base64(row[self.image_column])
        image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        text = self.processor.tokenizer(
            row[self.text_column],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        return image, text


def train_clip(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.train_csv)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    dataset = ImageTextDataset(df, processor, args.image_column, args.text_column)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.2,
    )
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        correct_image_to_text = 0
        correct_text_to_image = 0
        total_samples = 0
        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, texts in pbar:
            optimizer.zero_grad()
            images = images.to(device)
            texts = texts.to(device)

            outputs = model(pixel_values=images, input_ids=texts)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                image_to_text_preds = logits_per_image.argmax(dim=1)
                text_to_image_preds = logits_per_text.argmax(dim=1)
                correct_image_to_text += (image_to_text_preds == ground_truth).sum().item()
                correct_text_to_image += (text_to_image_preds == ground_truth).sum().item()
                total_samples += len(images)

            pbar.set_postfix(loss=loss.item())

        epoch_loss /= len(train_dataloader)
        img_to_txt_acc = correct_image_to_text / total_samples
        txt_to_img_acc = correct_text_to_image / total_samples
        epoch_accuracy = (img_to_txt_acc + txt_to_img_acc) / 2
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"Model saved to {args.model_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on image/text sentiment pairs.")
    parser.add_argument("--train-csv", default=DATASET_DIR / "training" / "train_data_post.csv", type=str)
    parser.add_argument("--model-output", default=MODEL_DIR / "clip_model.pth", type=str)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--image-column", default="filename")
    parser.add_argument("--text-column", default="post_mood_PNN")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    return parser.parse_args()


def main():
    train_clip(parse_args())


if __name__ == "__main__":
    main()
