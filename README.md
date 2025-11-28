# Sarcasm Detector

A web-based application that uses machine learning to detect sarcasm in text. Built with Flask and scikit-learn, featuring a modern, responsive UI.

## Features

- 🕵️ Real-time sarcasm detection
- 📊 Confidence score display
- 🎨 Modern, responsive web interface
- 💡 Example sentences for testing
- ⚡ Fast predictions using pre-trained ML model

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn, joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: TF-IDF Vectorization
- **Model**: Pre-trained classifier (SVM/Naive Bayes/etc.)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sarcasm-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure all model files are present:
   - `sarcasm_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `label_encoder.pkl`

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter any sentence in the text area and click "Analyze"

4. View the prediction result and confidence score

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── model.ipynb           # Jupyter notebook for model training
├── sarcasm2.csv          # Dataset used for training
├── templates/
│   └── index.html        # Web interface
├── sarcasm_model.pkl     # Trained ML model
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
└── label_encoder.pkl     # Label encoder
```

## Model Training

The model was trained using:
- Dataset: `sarcasm2.csv`
- Features: TF-IDF vectorization of text
- Algorithm: [Specify the algorithm used, e.g., SVM, Random Forest, etc.]

To retrain the model, run the `model.ipynb` notebook.

## API Endpoint

The application provides a REST API endpoint:

**POST** `/predict`

Request body:
```json
{
  "text": "Your text here"
}
```

Response:
```json
{
  "result": "😏 Sarcastic" | "😊 Not Sarcastic",
  "confidence": 0.85
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Specify license, e.g., MIT License]

## Acknowledgments

- Dataset source: [If applicable]
- Inspired by various NLP projects for sarcasm detection
