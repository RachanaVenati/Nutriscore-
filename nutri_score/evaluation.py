import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from nutri_score.paths import MODEL_DIR


def evaluate_predictions(csv_path, prediction_column, ground_truth_column, show_plot):
    df = pd.read_csv(csv_path)
    predictions = df[prediction_column]
    ground_truth = df[ground_truth_column]
    labels = ["Negative", "Neutral", "positive"]

    precision_per_class = precision_score(ground_truth, predictions, average=None, labels=labels)
    recall_per_class = recall_score(ground_truth, predictions, average=None, labels=labels)
    f1_per_class = f1_score(ground_truth, predictions, average=None, labels=labels)
    accuracy = accuracy_score(ground_truth, predictions)

    print("Metrics per class:")
    for i, label in enumerate(labels):
        print(f"Label: {label}")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1 Score: {f1_per_class[i]:.4f}")

    print(f"\nOverall Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    if show_plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment prediction CSV metrics.")
    parser.add_argument("--csv", default=MODEL_DIR / "test_data_post_with_predictions.csv", type=str)
    parser.add_argument("--prediction-column", default="predicted_post_sentiment")
    parser.add_argument("--ground-truth-column", default="post_mood_PNN_text")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_predictions(args.csv, args.prediction_column, args.ground_truth_column, args.show_plot)


if __name__ == "__main__":
    main()
