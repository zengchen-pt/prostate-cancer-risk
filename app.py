from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os                     # ← 必须导入 os，用来读取环境变量

app = Flask(__name__)
CORS(app)

# 加载训练好的模型（请务必把这个模型文件和 app.py 放在同一个文件夹里）
model = joblib.load("pca_screening_model_rf_calibrated.pkl")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # 如果有人在浏览器直接访问这个网址，就告诉他们服务是活的
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    # 接收前端发来的数据（JSON 格式）
    data = request.get_json()

    try:
        # ---- 重要：把 DRE 的中文字符串变成数字，不然模型会报错 ----
        # 这里的映射必须和训练模型时用的一模一样！
        # 如果训练时是：正常=0，可疑结节=1，质硬=2
        dre_mapping = {
            'normal': 0,
            'suspicious': 1,
            'hard': 2
        }
        if 'DRE' in data:
            data['DRE'] = dre_mapping.get(data['DRE'], 0)   # 没见过的值默认为 0（正常）

        # 把数据变成表格给模型预测
        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0, 1]   # 取出是“阳性”（高风险）的概率

        # 根据概率给一个风险等级和建议
        if proba < 0.4:
            level = '低风险'
            advice = 'AI建议：常规随访，每年复查PSA，关注症状变化。'
        elif proba < 0.7:
            level = '中风险'
            advice = 'AI建议：1~3个月内复查PSA，或结合多参数MRI进一步评估。'
        else:
            level = '高风险'
            advice = 'AI建议：尽快转诊泌尿外科，考虑前列腺穿刺活检。'

        # 把结果返回给前端页面
        return jsonify({
            'probability': round(float(proba), 4),
            'risk_level': level,
            'advice': advice
        })

    except Exception as e:
        # 如果出错了，告诉前端具体是什么错误
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # 关键修复：让 Render 来决定端口号
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)