# ================================================
# 🌐 Flask App - Smart Sarcasm Detector (with proper confidence score)
# ================================================

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re

app = Flask(__name__)

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Text cleaning + features (same as training)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    return text.strip()

def extract_features(text):
    cues = [
        "yeah right", "sure", "obviously", "totally", "great", "love", "perfect",
        "amazing", "as if", "just what i needed", "oh wonderful", "thanks a lot"
    ]
    cue_present = int(any(cue in text for cue in cues))
    contrast = int(any(w in text for w in ["not", "but", "though", "however", "although"]))
    return [cue_present, contrast]

def irony_features(text):
    pos_words = ['love', 'great', 'amazing', 'wonderful', 'happy', 'perfect', 'nice', 'enjoy']
    neg_words = ['boring', 'bad', 'hate', 'terrible', 'worst', 'ugly', 'awful', 'sad']
    polite = ['ok', 'fine', 'thanks', 'good']
    rude = ['stupid', 'bad', 'dumb', 'ugly', 'taste']

    pos_neg_mix = int(any(p in text for p in pos_words) and any(n in text for n in neg_words))
    polite_rude_mix = int(any(p in text for p in polite) and any(r in text for r in rude))
    self_neg = int('i' in text and any(w in text for w in ['not','don’t','never','no']))

    words = text.split()
    flip = 0
    for i in range(1,len(words)):
        if any(w in words[i-1] for w in pos_words) and any(w in words[i] for w in neg_words): flip=1; break
        if any(w in words[i-1] for w in neg_words) and any(w in words[i] for w in pos_words): flip=1; break

    return [pos_neg_mix, polite_rude_mix, self_neg, flip]

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        t = clean_text(text)
        text_vec = vectorizer.transform([t])
        feat = np.array(extract_features(t) + irony_features(t)).reshape(1, -1)
        full_input = np.hstack([text_vec.toarray(), feat])

        prob = model.predict_proba(full_input)[0]
        sarcasm_prob = float(prob[1])  # Probability for class 1 (Sarcastic)

        result = "Sarcastic 😏" if sarcasm_prob >= 0.5 else "Not Sarcastic 🙂"
        confidence = round(sarcasm_prob, 2)

        return jsonify({"result": result, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
