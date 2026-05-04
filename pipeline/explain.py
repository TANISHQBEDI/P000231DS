import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from transformers import AutoTokenizer

from aircraft_nlp.data.preprocessing import preprocess_dataframe
from aircraft_nlp.data.source import LocalFileSource
from aircraft_nlp.models.train import DistilBertClassifier

# Paths
MODEL_PATH = "models/model.pt"
TOKENIZER_PATH = "models/tokenizer"
LABEL_MAPPING_PATH = "models/label_mapping.json"
DATA_PATH = "data/raw/NLP_Dataset_2026_Expanded.xlsx"
OUTPUT_DIR = "models/inference"
SAMPLE_SIZE = 100
RANDOM_SEED = 42

def load_resources():
    with open(LABEL_MAPPING_PATH, "r") as f:
        raw_mapping = json.load(f)
    if all(str(k).isdigit() for k in raw_mapping.keys()):
        label_mapping = {int(k): v for k, v in raw_mapping.items()}
    else:
        label_mapping = {int(v): k for k, v in raw_mapping.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    model = DistilBertClassifier("distilbert-base-uncased", num_labels=len(label_mapping))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    
    return model, tokenizer, label_mapping


def predict(text, model, tokenizer, label_mapping):
    """Generates the prediction and confidence score."""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        confidence, predicted_label_idx = torch.max(probs, dim=-1)

    label_idx_int = int(predicted_label_idx.item())
    predicted_label = label_mapping.get(label_idx_int, f"Unknown (ID: {label_idx_int})")
    
    return predicted_label, confidence.item(), label_idx_int


def explain_prediction(text, model, tokenizer, label_mapping):
    """
    Explains the model prediction by evaluating all words 
    and listing their exact contribution to the decision.
    """
    # 1. Get prediction details
    predicted_label, confidence, label_idx = predict(text, model, tokenizer, label_mapping)
    
    # 2. Define the prediction function for SHAP
    def f(x):
        tv = tokenizer(x.tolist(), padding=True, truncation=True, max_length=192, return_tensors="pt")
        with torch.no_grad():
            logits = model(tv["input_ids"], tv["attention_mask"])
            return torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    # 3. Create the Explainer with a whitespace tokenizer mask
    masker = shap.maskers.Text(r"\s+")
    explainer = shap.Explainer(f, masker)
    shap_values = explainer([text])

    # 4. Extract tokens and SHAP values for the predicted class
    words = shap_values.data[0]
    contributions = shap_values.values[0][:, label_idx]

    # 5. Group and clean words
    word_contributions = {}
    for word, score in zip(words, contributions):
        clean_word = word.strip().strip(".,/-:").upper()
        
        # Filter out empty words, common punctuation, and boilerplate terms
        if not clean_word or clean_word in ["IAW", "AMM", "AND", "THE", "TO"]:
            continue
            
        word_contributions[clean_word] = word_contributions.get(clean_word, 0.0) + score

    # 6. Categorize the impact of EVERY word based on its score
    all_word_impacts = []
    for word, score in word_contributions.items():
        if score > 0.01:
            impact = "Strongly Supports"
        elif score > 0.0:
            impact = "Supports"
        elif score < -0.01:
            impact = "Strongly Contradicts"
        else:
            impact = "Contradicts"

        all_word_impacts.append({
            "word": word,
            "impact": impact,
            "raw_score": round(score, 4)
        })

    # Sort the list from the highest positive contributor down to the lowest
    all_word_impacts = sorted(all_word_impacts, key=lambda x: x['raw_score'], reverse=True)

    strong_words = [
        item["word"]
        for item in all_word_impacts
        if item["impact"] == "Strongly Supports"
    ]

    return {
        "text": text,
        "prediction": predicted_label,
        "confidence": round(confidence * 100, 2),
        "word_explanations": all_word_impacts,
        "strong_words": strong_words,
    }


def run_batch_inference():
    model, tokenizer, label_mapping = load_resources()

    source = LocalFileSource(DATA_PATH)
    df = preprocess_dataframe(source.load())
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)

    results = []
    actual_labels = []
    predicted_labels = []
    for _, row in df.iterrows():
        explanation = explain_prediction(row["text"], model, tokenizer, label_mapping)
        actual_labels.append(row["label"])
        predicted_labels.append(explanation["prediction"])
        results.append(
            {
                "text": row["text"],
                "actual_label": row["label"],
                "predicted_label": explanation["prediction"],
                "confidence": explanation["confidence"],
                "strong_words": explanation["strong_words"],
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"{timestamp}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")

    labels = sorted(set(actual_labels) | set(predicted_labels))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(actual_labels, predicted_labels):
        matrix[label_to_idx[actual], label_to_idx[predicted]] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{timestamp}_confusion_matrix.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {plot_path}")


if __name__ == "__main__":
    run_batch_inference()