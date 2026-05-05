from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sklearn
import traceback
import os

app = Flask(__name__)
CORS(app)

print(f"Scikit-learn version: {sklearn.__version__}")
model = joblib.load("pca_screening_model_rf_calibrated.pkl")
print("Model loaded successfully.")

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
        # 1. 将前端的 DRE 字符串转为数字
        if 'DRE' in data and isinstance(data['DRE'], str):
            dre_map = {'normal': 0, 'suspicious': 1, 'hard': 2}
            data['DRE'] = dre_map.get(data['DRE'], 0)

        # 2. 构建 DataFrame，确保列名和顺序与模型训练时完全一致
        df = pd.DataFrame([data])
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[EXPECTED_FEATURES]
        df = df.astype(np.float64).fillna(0)

        # 3. 关键步骤：显式重新包装为 DataFrame，防止被内部转换为 numpy 数组
        #    这是为了解决“仅支持 pandas DataFrames 使用字符串指定列”的错误
        df = pd.DataFrame(df, columns=EXPECTED_FEATURES)

        # 4. 预测
        proba = model.predict_proba(df)[0, 1]

        # 5. 风险分级
        if proba < 0.4:
            level, advice = '低风险', 'AI建议：常规随访，每年复查PSA，关注症状变化。'
        elif proba < 0.7:
            level, advice = '中风险', 'AI建议：1~3个月内复查PSA，或结合多参数MRI进一步评估。'
        else:
            level, advice = '高风险', 'AI建议：尽快转诊泌尿外科，考虑前列腺穿刺活检。'

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