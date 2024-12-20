import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# 读取数据
df = pd.read_csv('../exam_latest/dataset/illness/national_illness.csv')

# 转换date列为日期格式
df['date'] = pd.to_datetime(df['date'])

# 按时间排序
df = df.sort_values(by='date')

# 选择特征列
feature_keys = [
    "WEIGHTED_ILI",
    "UNWEIGHTED_ILI",
    "AGE_0-4",
    "AGE_5-24",
    "ILITOTAL",
    "NUM._OF_PROVIDERS"
]

# 目标列
target_key = 'OT'

# 标准化特征
scaler = StandardScaler()
features = df[feature_keys]
features_scaled = scaler.fit_transform(features)

# 目标列归一化
target_scaler = StandardScaler()
target = df[target_key]
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

# 数据划分，按时间划分为训练集和验证集
train_size = int(len(df) * 0.8)  # 前80%为训练集
train_features, val_features = features_scaled[:train_size], features_scaled[train_size:]
train_target, val_target = target_scaled[:train_size], target_scaled[train_size:]

# 创建时间序列数据集
def create_dataset(features, target, look_back=1):
    X, y = [], []
    for i in range(len(features) - look_back):  # 注意：len(features) - look_back
        # 检查目标值的索引是否超出范围
        if i + look_back < len(target):  # 确保目标索引不会越界
            X.append(features[i:i + look_back])
            y.append(target[i + look_back])  # 使用 `.iloc[]` 来通过位置索引
    return np.array(X), np.array(y)

# 设置回溯步长（例如，使用过去4周的数据来预测下周的OT）
look_back = 50

X_train, y_train = create_dataset(train_features, train_target, look_back)
X_val, y_val = create_dataset(val_features, val_target, look_back)

# 重塑输入数据为LSTM需要的形状
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(feature_keys)))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(feature_keys)))

# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 输出层
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# 模型训练
model = build_lstm_model((look_back, len(feature_keys)))
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 绘制损失曲线
def plot_loss_curve(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_loss_curve(history)

# 预测
y_pred = model.predict(X_val)

# 反归一化预测值
y_pred_rescaled = target_scaler.inverse_transform(y_pred)

# 反归一化真实值
y_val_rescaled = target_scaler.inverse_transform(y_val)

# 计算RMSE
def calculate_rmse(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    return rmse

rmse = calculate_rmse(y_val_rescaled, y_pred_rescaled)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# 可视化真实值与预测值的对比
plt.figure(figsize=(12, 6))
plt.plot(y_val_rescaled, label='True Values', color='blue')
plt.plot(y_pred_rescaled, label='Predicted Values', color='red')
plt.title('True vs Predicted OT Values')
plt.xlabel('Time')
plt.ylabel('OT')
plt.legend()
# 设置Y轴格式为正常数字，去掉科学计数法
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.show()
