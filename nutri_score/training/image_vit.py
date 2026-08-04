import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification

from nutri_score.image_utils import image_from_base64
from nutri_score.paths import DATASET_DIR, MODEL_DIR, PROJECT_ROOT

LABEL_TO_ID = {"positive": 0, "Negative": 1, "Neutral": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
MODEL_NAME = "google/vit-base-patch16-224"


class EmotionBase64Dataset(Dataset):
    def __init__(self, csv_file, image_column="filename", label_column="post_mood_PNN", transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = image_from_base64(row[self.image_column])
        if self.transform:
            image = self.transform(image)
        return image, LABEL_TO_ID[row[self.label_column]]


class ImageOnlyBase64Dataset(Dataset):
    def __init__(self, csv_file, image_column="filename", transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_column = image_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = image_from_base64(self.data.iloc[index][self.image_column])
        if self.transform:
            image = self.transform(image)
        return image


def default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def merge_train_validation(train_csv, val_csv, output_csv):
    merged_df = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], ignore_index=True)
    output_csv = PROJECT_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)
    return output_csv


def build_model(model_name):
    return ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )


def train_vit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_csv = args.train_csv
    if args.val_csv:
        train_csv = merge_train_validation(args.train_csv, args.val_csv, args.merged_output_csv)

    dataset = EmotionBase64Dataset(train_csv, args.image_column, args.label_column, default_transform())
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}], "
            f"Loss: {running_loss / len(train_loader):.4f}, "
            f"Accuracy: {100 * correct / total:.2f}%"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"Training complete. Model saved to {args.model_output}")


def predict_vit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name).to(device)
    model.load_state_dict(torch.load(args.model_output, map_location=device))
    model.eval()

    dataset = ImageOnlyBase64Dataset(args.test_csv, args.image_column, default_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images).logits
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())

    test_df = pd.read_csv(args.test_csv)
    test_df["predicted_post_mood_PNN"] = [ID_TO_LABEL[p] for p in predictions]
    output_csv = PROJECT_ROOT / args.predictions_output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run inference with a ViT image sentiment model.")
    parser.add_argument("--mode", choices=["train", "predict", "train-and-predict"], default="train-and-predict")
    parser.add_argument("--train-csv", default=DATASET_DIR / "training" / "train_data_post.csv", type=str)
    parser.add_argument("--val-csv", default=DATASET_DIR / "validation" / "validation_data_post.csv", type=str)
    parser.add_argument("--test-csv", default=DATASET_DIR / "testing" / "test_data_post.csv", type=str)
    parser.add_argument("--merged-output-csv", default="outputs/merged_file_train_img.csv")
    parser.add_argument("--predictions-output", default="outputs/test_emotions_with_predictions_7.csv")
    parser.add_argument("--model-output", default=MODEL_DIR / "emotion_vit_base64.pth", type=str)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--image-column", default="filename")
    parser.add_argument("--label-column", default="post_mood_PNN")
    parser.add_argument("--epochs", default=7, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode in {"train", "train-and-predict"}:
        train_vit(args)
    if args.mode in {"predict", "train-and-predict"}:
        predict_vit(args)


if __name__ == "__main__":
    main()
