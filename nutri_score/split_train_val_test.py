import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nutri_score.paths import DATASET_DIR


def split_dataset(input_csv, output_dir, random_state):
    original_df = pd.read_csv(input_csv)

    count_text_labels = original_df["post_mood_PNN"].value_counts()
    number_of_test_count = int(0.15 * len(original_df))
    samples_perclass = number_of_test_count // len(count_text_labels)

    print(f"Samples per class: {samples_perclass}")

    test_samples = pd.concat([
        original_df[original_df["post_mood_PNN"] == label].sample(samples_perclass, random_state=random_state)
        for label in count_text_labels.index
    ])

    train_val_samples = original_df.drop(test_samples.index)

    print("Test set distribution:")
    print(test_samples["post_mood_PNN"].value_counts())

    train_rows, val_rows = train_test_split(
        train_val_samples,
        test_size=0.176,
        random_state=random_state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows.to_csv(output_dir / "train_data_img.csv", index=False)
    val_rows.to_csv(output_dir / "validation_data_img.csv", index=False)
    test_samples.to_csv(output_dir / "test_data_img.csv", index=False)

    print(f"Files saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split annotated data into train, validation, and test CSVs.")
    parser.add_argument("--input-csv", default=DATASET_DIR / "kaggle" / "fully_annotated_dataset_cleaned_64.csv", type=str)
    parser.add_argument("--output-dir", default=DATASET_DIR / "splits", type=str)
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(args.input_csv, args.output_dir, args.random_state)


if __name__ == "__main__":
    main()
