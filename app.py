from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sklearn
import traceback

# 确保 monotonic_cst 存在（旧版 sklearn 本身就有，这行只是双保险）
from sklearn.tree import DecisionTreeClassifier
if not hasattr(DecisionTreeClassifier, 'monotonic_cst'):
    DecisionTreeClassifier.monotonic_cst = None

app = Flask(__name__)
CORS(app)
print(f"Scikit-learn version: {sklearn.__version__}")

model = joblib.load("pca_screening_model_rf_calibrated.pkl")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        dre_mapping = {'normal': 0, 'suspicious': 1, 'hard': 2}
        if 'DRE' in data:
            data['DRE'] = dre_mapping.get(data['DRE'], 0)

        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0, 1]

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
        print("Predict error:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)