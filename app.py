from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model, vectorizer, and label encoder
model = joblib.load("sarcasm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'})

        # Preprocess input
        text_vec = vectorizer.transform([text])
        
        # Predict
        pred_label_encoded = model.predict(text_vec)[0]
        pred_label = le.inverse_transform([pred_label_encoded])[0]

        # Get probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vec)[0]
            # Get probability for predicted class
            confidence = float(probs[pred_label_encoded])
        else:
            confidence = 0.8 if pred_label_encoded == 1 else 0.3  # fallback

        # Prepare result string
        result_text = "😏 Sarcastic" if pred_label_encoded == 1 else "😊 Not Sarcastic"

        return jsonify({'result': result_text, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
