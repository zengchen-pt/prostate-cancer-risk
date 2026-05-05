from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# 加载模型
model = joblib.load("pca_screening_model_rf_calibrated.pkl")

# ---------- 修复 scikit-learn 版本兼容性（针对 monotonic_cst 缺失）----------
def fix_monotonic_cst(estimator):
    """递归遍历模型中的决策树，补充缺失的 monotonic_cst 属性"""
    if hasattr(estimator, 'estimators_'):
        for sub_est in estimator.estimators_:
            if hasattr(sub_est, 'tree_') and not hasattr(sub_est, 'monotonic_cst'):
                sub_est.monotonic_cst = None
    if hasattr(estimator, 'calibrated_classifiers_'):
        for _, calibrated in estimator.calibrated_classifiers_.items():
            fix_monotonic_cst(calibrated.base_estimator)

fix_monotonic_cst(model)
# -----------------------------------------------------------

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)