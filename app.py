from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sklearn
from sklearn.tree import DecisionTreeClassifier
import traceback

# ---------- 强制补丁：在加载模型前添加 monotonic_cst ----------
# 确保即使是旧模型也能在新版 sklearn 上运行
if not hasattr(DecisionTreeClassifier, 'monotonic_cst'):
    DecisionTreeClassifier.monotonic_cst = None
    print("已为 DecisionTreeClassifier 添加 monotonic_cst 属性")

# 额外保险：给 DecisionTreeRegressor 也加上（虽不一定会用到）
from sklearn.tree import DecisionTreeRegressor
if not hasattr(DecisionTreeRegressor, 'monotonic_cst'):
    DecisionTreeRegressor.monotonic_cst = None
# -----------------------------------------------------------

app = Flask(__name__)
CORS(app)

print(f"Scikit-learn version: {sklearn.__version__}")

# 加载模型
model = joblib.load("pca_screening_model_rf_calibrated.pkl")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        # DRE 编码
        dre_mapping = {
            'normal': 0,
            'suspicious': 1,
            'hard': 2
        }
        if 'DRE' in data:
            data['DRE'] = dre_mapping.get(data['DRE'], 0)

        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0, 1]

        # 风险等级
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