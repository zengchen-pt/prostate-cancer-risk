from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sklearn
import sklearn.tree
import traceback

# ----- 关键修复：给 DecisionTreeClassifier 打补丁 -----
# 解决旧模型在新版 scikit-learn 下缺失 monotonic_cst 的问题
if not hasattr(sklearn.tree.DecisionTreeClassifier, 'monotonic_cst'):
    sklearn.tree.DecisionTreeClassifier.monotonic_cst = None
# ---------------------------------------------------

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
        # DRE 编码（将字符串转为数字）
        dre_mapping = {
            'normal': 0,
            'suspicious': 1,
            'hard': 2
        }
        if 'DRE' in data:
            data['DRE'] = dre_mapping.get(data['DRE'], 0)

        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0, 1]

        # 风险等级划分
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
        # 在 Render 日志中输出完整的错误信息
        print("Predict error:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)