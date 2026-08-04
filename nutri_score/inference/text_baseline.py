import argparse

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nutri_score.paths import DATASET_DIR, PROJECT_ROOT

LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "positive",
}
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def predict_sentiment(text, tokenizer, model):
    try:
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions).item()
        return LABEL_MAPPING[predicted_class]
    except Exception:
        return "Error"


def run_baseline(input_csv, output_csv, text_column, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    data = pd.read_csv(input_csv)
    output_csv = PROJECT_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data["predicted_sentiment"] = data[text_column].apply(lambda text: predict_sentiment(text, tokenizer, model))
    data.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline RoBERTa text sentiment inference.")
    parser.add_argument("--input-csv", default=DATASET_DIR / "testing" / "test_data_post.csv", type=str)
    parser.add_argument("--output-csv", default="outputs/predictions_baseline.csv")
    parser.add_argument("--text-column", default="body")
    parser.add_argument("--model-name", default=MODEL_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    run_baseline(args.input_csv, args.output_csv, args.text_column, args.model_name)


if __name__ == "__main__":
    main()
