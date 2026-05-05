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

# 根据模型训练时的特征顺序（从日志确认）
FEATURE_NAMES = [
    'age', 'tpsa', 'PV', 'PSAD', 'NLR', 'DRE',
    'BMI', 'hypertension', 'diabetes', 'hyperlipidemia', 'MRI'
]

# 手动指定哪些是类别特征（需要进行独热编码的列）
# 根据您的前端输入，这些列的值是 0/1/2 的整数，但本质是类别
CATEGORICAL_FEATURES = ['DRE', 'hypertension', 'diabetes', 'hyperlipidemia', 'MRI']

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        # 1. 转换 DRE 为数字（如果前端还是发了字符串）
        if 'DRE' in data and isinstance(data['DRE'], str):
            dre_map = {'normal': 0, 'suspicious': 1, 'hard': 2}
            data['DRE'] = dre_map.get(data['DRE'], 0)

        # 2. 构建原始 DataFrame，确保所有特征列都存在
        df = pd.DataFrame([data])
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0

        # 3. 区分处理数值特征和类别特征
        for col in FEATURE_NAMES:
            if col in CATEGORICAL_FEATURES:
                # 类别特征：转为整数，并限制在非负范围
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                # 数值特征：转为 float64
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float64)

        # 4. 统一列顺序
        df = df[FEATURE_NAMES]

        # 5. 显式重新包装为 DataFrame，保留列名（老规矩）
        df = pd.DataFrame(df, columns=FEATURE_NAMES)

        # 6. 预测
        proba = model.predict_proba(df)[0, 1]

        # 7. 风险分级
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