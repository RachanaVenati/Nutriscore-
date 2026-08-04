import argparse

import pandas as pd
import requests

from nutri_score.image_utils import strip_data_url_prefix
from nutri_score.paths import DATASET_DIR, PROJECT_ROOT

DEFAULT_PROMPT = """Give me the sentiment of the image, i.e., which emotion is evoked upon seeing the image sent to you.

Definitions:
- POSITIVE: The image evokes happiness, joy, warmth, excitement, or inspiration, sarcasm, comics, animation, animals, smiling faces, natural beauty, celebrations, art, fashion, photography, wildlife, or cozy settings.
- NEGATIVE: The image evokes sadness, fear, discomfort, distress, or danger. It may include blood, weapons, crying faces, destruction, loneliness, or threatening situations, depression, suicidal themes, or crime.
- NEUTRAL: The image does not evoke strong positive or negative emotions. It includes news-related images, technical images, or plain landscapes with balanced elements.

You should respond only in three labels: POSITIVE, NEGATIVE, or NEUTRAL.

DO NOT GIVE ME REASONING, JUST GIVE A SINGLE WORD RESPONSE.
Always respond in lowercase.
"""


def llava_request(api_url, prompt, base64_image, model_name, timeout):
    data = {
        "model": model_name,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "num_ctx": 32768,
            "temperature": 0.0,
        },
    }
    response = requests.post(api_url, json=data, timeout=timeout)
    if response.status_code == 200:
        return response.json()["response"]
    print(f"Error: {response.status_code} - {response.text}")
    return None


def clean_input_csv(input_csv, cleaned_csv, image_column):
    df = pd.read_csv(input_csv)
    df[image_column] = df[image_column].apply(strip_data_url_prefix)
    cleaned_csv = PROJECT_ROOT / cleaned_csv
    cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_csv, index=False)
    return cleaned_csv


def process_csv(input_csv, output_csv, api_url, model_name, image_column, prompt, timeout):
    df = pd.read_csv(input_csv)
    if "llava_response" not in df.columns:
        df["llava_response"] = ""

    results = []
    for index, row in df.iterrows():
        try:
            sentiment = llava_request(api_url, prompt, row[image_column], model_name, timeout)
            df.at[index, "llava_response"] = sentiment
            results.append({"sentiment_score": sentiment})
        except KeyError as e:
            print(f"Skipping row {index} due to missing key: {e}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    output_csv = PROJECT_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaVA image sentiment inference through an Ollama-compatible API.")
    parser.add_argument("--input-csv", default=DATASET_DIR / "testing" / "test_data_post.csv", type=str)
    parser.add_argument("--cleaned-csv", default="outputs/cleaned_b64file.csv")
    parser.add_argument("--output-csv", default="outputs/llava_img_senti.csv")
    parser.add_argument("--api-url", default="http://gammaweb05.medien.uni-weimar.de:11439/api/generate")
    parser.add_argument("--model-name", default="llava:latest")
    parser.add_argument("--image-column", default="filename")
    parser.add_argument("--timeout", default=120, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    cleaned_csv = clean_input_csv(args.input_csv, args.cleaned_csv, args.image_column)
    process_csv(
        cleaned_csv,
        args.output_csv,
        args.api_url,
        args.model_name,
        args.image_column,
        DEFAULT_PROMPT,
        args.timeout,
    )


if __name__ == "__main__":
    main()
