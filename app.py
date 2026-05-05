from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sklearn
import traceback
import os  # 必须导入，用于读取环境变量 PORT

app = Flask(__name__)
CORS(app)

print(f"Scikit-learn version: {sklearn.__version__}")

# 加载模型
model = joblib.load("pca_screening_model_rf_calibrated.pkl")
print("Model loaded successfully.")

# 获取模型期望的特征名
expected_features = [
    'age', 'tpsa', 'PV', 'PSAD', 'NLR', 'DRE',
    'BMI', 'hypertension', 'diabetes', 'hyperlipidemia', 'MRI'
]
if hasattr(model, 'feature_names_in_'):
    expected_features = list(model.feature_names_in_)
    print(f"Model feature_names_in_: {expected_features}")
else:
    print(f"Using fallback feature names: {expected_features}")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        # 1. 将 DRE 转为数字（兼容前端直接发送字符串的情况）
        if 'DRE' in data and isinstance(data['DRE'], str):
            dre_map = {'normal': 0, 'suspicious': 1, 'hard': 2}
            data['DRE'] = dre_map.get(data['DRE'], 0)

        # 2. 构建 DataFrame，确保所有需要的列都存在
        df = pd.DataFrame([data])
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0  # 缺失列用 0 填充

        # 3. 强制将所有列转为数值类型
        #    先将所有值转为 float，无法转换的变为 NaN
        df = df[expected_features].apply(pd.to_numeric, errors='coerce')
        #    将所有 NaN 填充为 0
        df.fillna(0, inplace=True)
        #    最终确保整个 DataFrame 的数据类型为 float64
        df = df.astype(np.float64)

        # 4. 诊断日志（可在 Render 控制台查看）
        print("DataFrame dtypes:\n", df.dtypes)
        print("DataFrame values:\n", df.values)

        # 5. 预测
        proba = model.predict_proba(df)[0, 1]

        # 6. 风险分级
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