from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sklearn
import traceback
import os

app = Flask(__name__)
CORS(app)

print(f"Scikit-learn version: {sklearn.__version__}")

model = joblib.load("pca_screening_model_rf_calibrated.pkl")
print("Model loaded successfully.")

# 模型期望的特征顺序（从日志获得）
EXPECTED_FEATURES = [
    'age', 'tpsa', 'PV', 'PSAD', 'NLR', 'DRE',
    'BMI', 'hypertension', 'diabetes', 'hyperlipidemia', 'MRI'
]

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        # 1. 确保 DRE 是数字（前端可能发字符串）
        if 'DRE' in data and isinstance(data['DRE'], str):
            dre_map = {'normal': 0, 'suspicious': 1, 'hard': 2}
            data['DRE'] = dre_map.get(data['DRE'], 0)

        # 2. 严格按照模型期望的顺序，提取数值并构建 NumPy 数组
        feature_values = []
        for col in EXPECTED_FEATURES:
            val = data.get(col, 0)          # 缺失的列填 0
            try:
                val = float(val)            # 强制转为浮点数
            except (ValueError, TypeError):
                val = 0.0
            feature_values.append(val)
        
        X = np.array([feature_values], dtype=np.float64)

        # 可选：打印数组供日志查看
        print("Input array:", X)

        # 3. 直接用数组预测
        proba = model.predict_proba(X)[0, 1]

        # 4. 风险等级
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