from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("pca_screening_model_rf_calibrated.pkl")

# ---------- 更彻底的 monotonic_cst 修补 ----------
def add_monotonic_cst(obj, visited=None):
    """
    递归遍历对象的所有属性，为所有具有 tree_ 属性的对象
    （即 DecisionTreeClassifier 实例）添加 monotonic_cst 属性
    """
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # 如果对象本身就有 tree_ 属性，说明是决策树，直接设置
    if hasattr(obj, 'tree_') and not hasattr(obj, 'monotonic_cst'):
        try:
            obj.monotonic_cst = None
        except Exception:
            pass

    # 遍历对象的属性
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if isinstance(attr_value, (list, tuple, set)):
                for item in attr_value:
                    if hasattr(item, '__dict__'):
                        add_monotonic_cst(item, visited)
            elif hasattr(attr_value, '__dict__'):
                add_monotonic_cst(attr_value, visited)

    # 处理列表、字典等容器
    if isinstance(obj, dict):
        for key, val in obj.items():
            if hasattr(val, '__dict__'):
                add_monotonic_cst(val, visited)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            if hasattr(item, '__dict__'):
                add_monotonic_cst(item, visited)

# 执行修补
add_monotonic_cst(model)
# --------------------------------------------

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