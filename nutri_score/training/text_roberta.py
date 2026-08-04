import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from nutri_score.paths import DATASET_DIR, MODEL_DIR, PROJECT_ROOT

LABEL_TO_ID = {"positive": 0, "Neutral": 1, "Negative": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _load_dataset(csv_path, tokenizer):
    from datasets import Dataset

    df = pd.read_csv(csv_path).rename(columns={"post_mood_PNN_text": "labels"})
    df["labels"] = df["labels"].map(LABEL_TO_ID)

    def tokenize_function(examples):
        examples["body"] = [str(text) if text else "" for text in examples["body"]]
        return tokenizer(examples["body"], padding="max_length", truncation=True, max_length=512)

    return df, Dataset.from_pandas(df).map(tokenize_function, batched=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train_text_model(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    output_dir = PROJECT_ROOT / args.output_dir
    logging_dir = PROJECT_ROOT / args.logging_dir
    model_output_dir = MODEL_DIR / "roberta_finetuned_sentiment"
    weights_output_path = MODEL_DIR / "roberta_finetuned.pth"
    predictions_output_path = PROJECT_ROOT / args.predictions_output

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df, train_dataset = _load_dataset(args.train_csv, tokenizer)
    _, val_dataset = _load_dataset(args.val_csv, tokenizer)
    test_df, test_dataset = _load_dataset(args.test_csv, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        evaluation_strategy="epoch",
        logging_dir=str(logging_dir),
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(model_output_dir))
    torch.save(model.state_dict(), weights_output_path)

    results = trainer.evaluate(test_dataset)
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(axis=-1)

    output_df = pd.read_csv(args.test_csv)
    output_df["predicted_label"] = [ID_TO_LABEL[label] for label in predicted_labels]
    for key, value in results.items():
        output_df[key] = value
    output_df.to_csv(predictions_output_path, index=False)

    print(f"Model saved to {model_output_dir}")
    print(f"Weights saved to {weights_output_path}")
    print(f"Predictions saved to {predictions_output_path}")
    return train_df, test_df


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for text sentiment classification.")
    parser.add_argument("--train-csv", default=DATASET_DIR / "training" / "train_data_post.csv", type=str)
    parser.add_argument("--val-csv", default=DATASET_DIR / "validation" / "validation_data_post.csv", type=str)
    parser.add_argument("--test-csv", default=DATASET_DIR / "testing" / "test_data_post.csv", type=str)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--logging-steps", default=100, type=int)
    parser.add_argument("--run-name", default="roberta-text")
    parser.add_argument("--output-dir", default="outputs/text_roberta/results")
    parser.add_argument("--logging-dir", default="outputs/text_roberta/logs")
    parser.add_argument("--predictions-output", default="outputs/updated_test_data_with_predictions.csv")
    return parser.parse_args()


def main():
    train_text_model(parse_args())


if __name__ == "__main__":
    main()
