from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ViTModel

from nutri_score.paths import MODEL_DIR

TEXT_MODEL_PATH = MODEL_DIR / "roberta_finetuned.pth"
VIT_MODEL_PATH = MODEL_DIR / "emotion_vit_base64.pth"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_sentiment_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
text_model_cls = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    num_labels=3,
)
text_model_cls.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
text_model_cls.to(device)
text_model = text_model_cls.roberta
text_model.eval()

vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
vit_model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device), strict=False)
vit_model.to(device)
vit_model.eval()


class FusionModel(nn.Module):
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


fusion_model = FusionModel()
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
fusion_model.to(device)
fusion_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

inverse_label_map = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
}

app = FastAPI()


@app.post("/predict-sentiment/")
async def predict_sentiment(body: str = Form(...), file: UploadFile = File(...)):
    try:
        inputs = text_tokenizer(body, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_emb = text_model(**inputs).last_hidden_state[:, 0, :]

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_emb = vit_model(image_tensor).last_hidden_state[:, 0, :]

        with torch.no_grad():
            output = fusion_model(text_emb, image_emb)
            pred = torch.argmax(output, dim=1).item()
            sentiment = inverse_label_map[pred]

        return {"sentiment": sentiment}

    except Exception as e:
        return {"error": str(e)}


def main():
    uvicorn.run("nutri_score.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
