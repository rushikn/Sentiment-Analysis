from flask import Flask, request, jsonify
from flask_cors import CORS
from model.sentiment_model import analyze_sentiment

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}) 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment, confidence = analyze_sentiment(text)
    return jsonify({"sentiment": sentiment, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000) 
