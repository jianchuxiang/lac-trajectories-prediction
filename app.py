import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
import os

# 初始化Flask应用
app = Flask(__name__)

# --- 在应用启动时，加载模型、预处理器和特征列表 ---
try:
    # 加载CatBoost模型文件
    model = joblib.load('catboost_model.joblib')
    
    # 加载预处理器和特征列表文件
    scaler = joblib.load('scaler.joblib')
    top_features, numerical_features = joblib.load('top_features.joblib')
    
    # 动态计算特征数量
    num_features = len(top_features)

    print("模型和预处理器加载成功！")
    print(f"模型期望的特征 ({num_features}个): {top_features}")

except FileNotFoundError as e:
    print(f"错误：找不到必要的模型文件: {e}")
    print("请确保 'catboost_model.joblib', 'scaler.joblib', 和 'top_features.joblib' 文件都在主目录中。")
    model, scaler, top_features, numerical_features, num_features = None, None, None, None, 0
except Exception as e:
    print(f"加载过程中发生未知错误: {e}")
    model, scaler, top_features, numerical_features, num_features = None, None, None, None, 0


# --- 创建一个路由来渲染主页 ---
# 这个路由会读取特征列表，并将其传递给HTML模板
@app.route('/')
def home():
    if not top_features:
        # 如果模型文件加载失败，显示一个错误页面或信息
        return "<h1>Error: Model files not loaded.</h1><p>Please check the server logs for more details.</p>", 500
    
    # 渲染HTML页面，并将特征数量传递给它，用于动态显示标题
    return render_template('index.html', num_features=num_features)


# --- 创建一个用于预测的API路由 ---
@app.route('/predict', methods=['POST'])
def predict():
    # 检查模型是否已成功加载
    if not all([model, scaler, top_features]):
        return jsonify({'error': '模型未加载，无法进行预测'}), 500

    try:
        # 1. 从POST请求中获取JSON格式的用户输入数据
        data = request.get_json(force=True)
        print(f"接收到的原始数据: {data}")

        # 2. 将输入数据转换为DataFrame，并【严格】按照训练时的特征顺序排列
        # 这一点至关重要，能防止因用户输入顺序不同导致的错误
        input_df = pd.DataFrame([data])
        input_df = input_df[top_features]

        # 3. 使用【已加载】的Scaler来转换数值特征
        # 注意：只转换在训练时被识别为数值特征的列
        if numerical_features:
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        print(f"标准化后的数据:\n{input_df.to_string()}")

        # 4. 使用加载的模型进行预测
        # .predict_proba() 返回一个二维数组，[[概率0, 概率1]]
        # 我们需要的是类别为1的概率，所以取 [:, 1]
        prediction_proba = model.predict_proba(input_df)[:, 1]
        
        # .predict() 返回预测的类别 (0 或 1)
        prediction = model.predict(input_df)

        # 5. 准备要返回给前端的JSON结果
        # 【重要更新】将预测结果文本修改为纯英文
        result = {
            'prediction': 'Risk' if prediction[0] == 1 else 'Not Risk',
            'probability_of_risk': round(float(prediction_proba[0]), 4)
        }
        print(f"预测结果: {result}")

        # 6. 将结果以JSON格式返回
        return jsonify(result)

    except Exception as e:
        # 捕获任何可能发生的错误，并返回一个清晰的错误信息
        print(f"预测过程中出错: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 400

# --- 运行Flask应用 ---
# 当直接运行这个脚本时，启动Web服务器
if __name__ == '__main__':
    # host='0.0.0.0' 使其可以被局域网内的其他设备访问
    # port=5000 是Flask默认的端口号
    # debug=True 可以在开发时看到详细的错误信息，在生产环境中应设为False
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

