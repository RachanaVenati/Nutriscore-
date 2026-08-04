import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ViTModel

from nutri_score.image_utils import image_from_base64
from nutri_score.paths import DATASET_DIR, MODEL_DIR

TEXT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
VIT_MODEL_NAME = "google/vit-base-patch16-224"
LABEL_TO_ID = {"positive": 0, "Negative": 1, "Neutral": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class FusionClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=512, num_labels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, text_emb, image_emb):
        x = torch.cat((text_emb, image_emb), dim=1)
        return self.fc(x)


class FusionSentimentDataset(Dataset):
    def __init__(self, csv_file, text_tokenizer, text_model, vit_model, device, label_column="label_post"):
        self.data = pd.read_csv(csv_file)
        self.text_tokenizer = text_tokenizer
        self.text_model = text_model
        self.vit_model = vit_model
        self.device = device
        self.label_column = label_column
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_emb, image_emb = extract_embeddings(
            str(row["body"]),
            row["filename"],
            self.text_tokenizer,
            self.text_model,
            self.vit_model,
            self.transform,
            self.device,
        )
        label = LABEL_TO_ID[row[self.label_column]]
        return text_emb.squeeze(0), image_emb.squeeze(0), torch.tensor(label)


def load_embedding_models(args, device):
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    text_model_cls = AutoModelForSequenceClassification.from_pretrained(args.text_model_name, num_labels=3)
    text_model_cls.load_state_dict(torch.load(args.text_weights, map_location=device))
    text_model_cls.to(device).eval()
    text_model = text_model_cls.roberta

    vit_model = ViTModel.from_pretrained(args.vit_model_name)
    vit_model.load_state_dict(torch.load(args.vit_weights, map_location=device), strict=False)
    vit_model.to(device).eval()
    return text_tokenizer, text_model, vit_model


def extract_embeddings(text, image_base64, text_tokenizer, text_model, vit_model, transform, device):
    encoded_input = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    with torch.no_grad():
        text_emb = text_model(**encoded_input).last_hidden_state[:, 0, :]

    image = image_from_base64(image_base64)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = vit_model(image_tensor).last_hidden_state[:, 0, :]
    return text_emb, image_emb


def train_fusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_tokenizer, text_model, vit_model = load_embedding_models(args, device)
    train_data = FusionSentimentDataset(args.train_csv, text_tokenizer, text_model, vit_model, device, args.label_column)
    val_data = FusionSentimentDataset(args.val_csv, text_tokenizer, text_model, vit_model, device, args.label_column)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    model = FusionClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for text_emb, image_emb, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(text_emb.to(device), image_emb.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for text_emb, image_emb, labels in val_loader:
                outputs = model(text_emb.to(device), image_emb.to(device))
                preds = outputs.argmax(dim=1)
                correct += (preds.cpu() == labels).sum().item()
                total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{args.epochs}, "
            f"Train Loss: {total_loss / len(train_loader):.4f}, "
            f"Val Acc: {100 * correct / total:.2f}%"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.fusion_weights)
    print(f"Fusion model saved at {args.fusion_weights}")


def predict_fusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_tokenizer, text_model, vit_model = load_embedding_models(args, device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    fusion_model = FusionClassifier().to(device)
    fusion_model.load_state_dict(torch.load(args.fusion_weights, map_location=device))
    fusion_model.eval()

    test_df = pd.read_csv(args.test_csv)
    predicted_labels = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            text_emb, image_emb = extract_embeddings(
                str(row["body"]),
                row["filename"],
                text_tokenizer,
                text_model,
                vit_model,
                transform,
                device,
            )
            output = fusion_model(text_emb, image_emb)
            pred = torch.argmax(output, dim=1).item()
            predicted_labels.append(ID_TO_LABEL[pred])

    test_df["predicted_post_sentiment"] = predicted_labels
    test_df.to_csv(args.predictions_output, index=False)
    print(f"Predictions saved to {args.predictions_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run inference with the RoBERTa + ViT fusion model.")
    parser.add_argument("--mode", choices=["train", "predict", "train-and-predict"], default="train-and-predict")
    parser.add_argument("--train-csv", default=DATASET_DIR / "training" / "train_data_post.csv", type=str)
    parser.add_argument("--val-csv", default=DATASET_DIR / "validation" / "validation_data_post.csv", type=str)
    parser.add_argument("--test-csv", default=DATASET_DIR / "testing" / "test_data_post.csv", type=str)
    parser.add_argument("--predictions-output", default=MODEL_DIR / "test_data_post_with_predictions.csv", type=str)
    parser.add_argument("--text-weights", default=MODEL_DIR / "roberta_finetuned.pth", type=str)
    parser.add_argument("--vit-weights", default=MODEL_DIR / "emotion_vit_base64.pth", type=str)
    parser.add_argument("--fusion-weights", default=MODEL_DIR / "fusion_sentiment_model.pth", type=str)
    parser.add_argument("--text-model-name", default=TEXT_MODEL_NAME)
    parser.add_argument("--vit-model-name", default=VIT_MODEL_NAME)
    parser.add_argument("--label-column", default="label_post")
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode in {"train", "train-and-predict"}:
        train_fusion(args)
    if args.mode in {"predict", "train-and-predict"}:
        predict_fusion(args)


if __name__ == "__main__":
    main()
