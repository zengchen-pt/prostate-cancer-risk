from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sklearn
import traceback
import sys

app = Flask(__name__)
CORS(app)

# 1. 打印 scikit-learn 版本（方便您从日志中确认）
print(f"Scikit-learn version: {sklearn.__version__}")

# 2. 加载模型
try:
    model = joblib.load("pca_screening_model_rf_calibrated.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Failed to load model: {e}")
    traceback.print_exc()
    sys.exit(1)

# 3. 自动获取模型期望的特征名列表
def get_model_feature_names(model):
    """尝试从模型中提取训练时使用的特征名"""
    # 方式1：某些 sklearn 模型直接保存了 feature_names_in_ 属性
    if hasattr(model, 'feature_names_in_'):
        print(f"Model feature_names_in_: {list(model.feature_names_in_)}")
        return list(model.feature_names_in_)
    
    # 方式2：如果是 Pipeline，尝试从第一步获取特征名
    if hasattr(model, 'named_steps'):
        steps = model.named_steps
        print(f"Pipeline steps: {list(steps.keys())}")
        first_step = steps[list(steps.keys())[0]]
        if hasattr(first_step, 'feature_names_in_'):
            print(f"First step feature_names_in_: {list(first_step.feature_names_in_)}")
            return list(first_step.feature_names_in_)
        # 如果是 ColumnTransformer，可以用 get_feature_names_out
        if hasattr(first_step, 'get_feature_names_out'):
            try:
                # 需要提供一个示例数据，这里用 dummy DataFrame 可能会失败，故返回 None
                pass
            except:
                pass
    return None

expected_features = get_model_feature_names(model)
if expected_features is None:
    # 如果无法自动获取，使用您之前的猜测（可根据日志调整）
    expected_features = [
        'age', 'tpsa', 'PV', 'PSAD', 'NLR', 'BMI',
        'hypertension', 'diabetes', 'hyperlipidemia', 'MRI', 'DRE'
    ]
    print(f"Using fallback feature names: {expected_features}")
else:
    print(f"Using model-provided feature names: {expected_features}")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    data = request.get_json()
    try:
        # DRE 字符串转数字（前端也可能已经转了，但后端保险处理）
        if 'DRE' in data and isinstance(data['DRE'], str):
            dre_mapping = {'normal': 0, 'suspicious': 1, 'hard': 2}
            data['DRE'] = dre_mapping.get(data['DRE'], 0)

        # 创建 DataFrame，并只保留模型期望的特征，按期望顺序排列
        df = pd.DataFrame([data])
        # 确保所有需要的列都存在，缺失的补 0
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        # 按模型期望的顺序选择列
        df = df[expected_features].copy()
        # 强制转为数值类型，无法转换的变成 NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # 填充 NaN
        df.fillna(0, inplace=True)
        df = df.astype(np.float64)

        # 打印调试信息（您在 Render 日志中能看到）
        print(f"Request data keys: {list(data.keys())}")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"DataFrame dtypes:\n{df.dtypes}")
        print(f"DataFrame values:\n{df.values}")

        # 预测
        proba = model.predict_proba(df)[0, 1]

        # 风险分级
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