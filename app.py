from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)
model = joblib.load("pca_screening_model_v2.pkl")

@app.route('/predict', methods=['GET', 'POST'])  # 👈 关键修改：允许GET和POST两种请求
def predict():
    # 场景 1：如果是GET请求（比如浏览器访问或UptimeRobot监控），返回服务状态
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Service is alive'})

    # 场景 2：如果是POST请求，执行AI预测
    data = request.get_json()
    try:
        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0, 1]
        return jsonify({
            'probability': round(float(proba), 4),
            'risk_level': '高风险' if proba > 0.7 else ('中风险' if proba > 0.3 else '低风险')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
