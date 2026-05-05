from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)
model = joblib.load("pca_screening_model_rf_calibrated.pkl")  # 使用新模型

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0, 1]

        # 调整后的风险等级（更贴合筛查场景）
        if proba < 0.4:
            level = '低风险'
            advice = 'AI建议：常规随访，每年复查PSA，关注症状变化。'
        elif proba < 0.7:
            level = '中风险'
            advice = 'AI建议：1~3个月内复查PSA，或结合多参数MRI进一步评估。'
        else:
            level = '高风险'
            advice = 'AI建议：尽快转诊泌尿外科，考虑前列腺穿刺活检。'

        return jsonify({
            'probability': round(float(proba), 4),
            'risk_level': level,
            'advice': advice
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
