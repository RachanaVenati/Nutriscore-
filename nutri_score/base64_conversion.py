import argparse
import base64
from pathlib import Path

import pandas as pd


def image_to_data_url(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
            image_type = image_path.split(".")[-1]
            return f"data:image/{image_type};base64,{base64_string}"
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def convert_images_in_csv(csv_path, image_column, output_csv_path):
    df = pd.read_csv(csv_path)
    df[image_column] = df[image_column].apply(image_to_data_url)

    df.to_csv(output_csv_path, index=False)
    print(f"Saved data URL CSV to {output_csv_path}")


def merge_metadata(source_csv, metadata_csv, image_dir, merged_output_csv):
    df1 = pd.read_csv(source_csv)
    df2 = pd.read_csv(metadata_csv)
    df2.rename(columns={"url": "image_url"}, inplace=True)
    merged_df = pd.merge(df1, df2[["image_url", "filename"]], on="image_url", how="left")
    image_dir = Path(image_dir)
    merged_df["filename"] = merged_df["filename"].apply(lambda filename: str(image_dir / filename))
    merged_df.to_csv(merged_output_csv, index=False)
    return merged_output_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge image metadata with a CSV and convert image paths to base64 data URLs."
    )
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--merged-output-csv", required=True)
    parser.add_argument("--base64-output-csv", required=True)
    parser.add_argument("--image-column", default="filename")
    return parser.parse_args()


def main():
    args = parse_args()
    merged_csv = merge_metadata(args.source_csv, args.metadata_csv, args.image_dir, args.merged_output_csv)
    convert_images_in_csv(merged_csv, args.image_column, args.base64_output_csv)


if __name__ == "__main__":
    main()
