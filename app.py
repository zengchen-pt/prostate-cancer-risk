from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sklearn
import traceback

app = Flask(__name__)
CORS(app)

print(f"Scikit-learn version: {sklearn.__version__}")

# 加载模型
model = joblib.load("pca_screening_model_rf_calibrated.pkl")
print("Model loaded successfully.")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        # 1. 将 DRE 转换为数字
        dre_mapping = {'normal': 0, 'suspicious': 1, 'hard': 2}
        if 'DRE' in data:
            data['DRE'] = dre_mapping.get(data['DRE'], 0)

        # 2. 创建 DataFrame
        df = pd.DataFrame([data])

        # 3. 【核心修复】确保数据类型正确，将可能包含 None 的列转为数值
        # 首先，确保所有列名都是字符串，避免数字列名干扰
        df.columns = df.columns.astype(str)
        
        # 第二步，将所有看起来像数字的列强制转为 float64
        for col in df.columns:
            # 尝试将列转换为数值，无法转换的设为 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 第三步，用 0 填充所有 NaN 值，避免模型报错
        df = df.fillna(0)

        print(f"Data types after cleaning:\n{df.dtypes}")
        print(f"Data values:\n{df.values}")

        # 4. 模型预测
        proba = model.predict_proba(df)[0, 1]

        # 5. 风险分级
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