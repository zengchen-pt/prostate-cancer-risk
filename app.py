from flask import Flask, request, jsonify
from flask_cors import CORS                     # ← 新增
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)                                      # ← 新增：允许所有来源的跨域请求
model = joblib.load("pca_screening_model_v2.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0, 1]
        return jsonify({
            'probability': round(float(proba), 4),
            'risk_level': '高风险' if proba > 0.7 else ('中风险' if proba > 0.3 else '低风险')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)   # ← 建议保留 debug=False