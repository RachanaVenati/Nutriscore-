# Nutri-Score Multimodal Sentiment Analysis

Nutri-Score is a multimodal sentiment analysis project for Instagram posts. It predicts whether a post is `positive`, `Neutral`, or `Negative` by using both the post text and the post image.

The project includes text-only, image-only, CLIP-based, and image-text fusion experiments, plus a FastAPI endpoint for serving the trained fusion model.

## Public Dataset And Models

The dataset and trained models are available publicly:

- Dataset: [Annotated Instagram Posts for Sentiment Analysis](https://www.kaggle.com/datasets/rachanavenati/annotated-instagram-posts-for-sentiment-analysis)
- Fusion model: [image_text_fusion_for_insta_posts_sentiment](https://huggingface.co/RachanaV/image_text_fusion_for_insta_posts_sentiment)
- Image model: [VIT_model_instagram_posts_sentiment](https://huggingface.co/RachanaV/VIT_model_instagram_posts_sentiment)
- Text model: [Finetuned_Roberta_Instagram_posts](https://huggingface.co/RachanaV/Finetuned_Roberta_Instagram_posts)

## Project Structure

```text
nutri_score/              Python package with training, inference, evaluation, and API code
annotation_templates/     Label Studio annotation template
docs/notebooks_archive/   Archived exploratory notebooks
docs/presentations/       Project presentation files
pyproject.toml            Package metadata, dependencies, and CLI entry points
```

Large artifacts are intentionally excluded from GitHub:

```text
dataset/                  Local dataset CSVs downloaded from Kaggle
models/                   Local model weights downloaded from Hugging Face
outputs/                  Training outputs
logs/                     Local logs
wandb/                    Weights & Biases runs
```

## Reproducibility

This project is reproducible, but not from the GitHub repository alone. To reproduce the training and inference workflow, clone the repository, install the package, then download the dataset from Kaggle and the trained model weights from Hugging Face.

Expected dataset paths:

```text
dataset/training/train_data_post.csv
dataset/validation/validation_data_post.csv
dataset/testing/test_data_post.csv
dataset/kaggle/fully_annotated_dataset_cleaned_64.csv
```

Expected model paths for API inference:

```text
models/roberta_finetuned.pth
models/emotion_vit_base64.pth
models/fusion_sentiment_model.pth
```

If the downloaded Hugging Face files use different names, rename or copy them into the filenames expected above.

The dataset should include post text in `body`, image data in `filename` or `insta_post_img_base64`, and sentiment labels such as `post_mood_PNN`, `post_mood_PNN_text`, or `label_post`, depending on the script.

## Setup

Use Python 3.8 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The scripts use pretrained models from Hugging Face:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
google/vit-base-patch16-224
openai/clip-vit-base-patch32
```

## Commands

Split the annotated dataset:

```bash
nutri-score-split \
  --input-csv dataset/kaggle/fully_annotated_dataset_cleaned_64.csv \
  --output-dir dataset/splits
```

Train the text model:

```bash
nutri-score-train-text
```

Train the image model:

```bash
nutri-score-train-vit --image-column filename
```

Train the CLIP model:

```bash
nutri-score-train-clip
```

Train the fusion model:

```bash
nutri-score-fusion
```

Evaluate predictions:

```bash
nutri-score-evaluate \
  --csv models/test_data_post_with_predictions.csv \
  --prediction-column predicted_post_sentiment \
  --ground-truth-column post_mood_PNN_text
```

Run the API after restoring the trained weights:

```bash
nutri-score-api
```

Prediction endpoint:

```text
POST http://localhost:8000/predict-sentiment/
```

Form fields:

```text
body    Instagram post text
file    Instagram post image
```
