# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
#
# # 设置字体为支持中文的字体
# plt.rcParams['font.family'] = 'SimHei'  # 如果 SimHei 没有，尝试 'Microsoft YaHei'
#
# # 预处理函数：清理列名、标准化数据、准备特征和目标变量
# def preprocess_data(file_path):
#     # 读取数据
#     data = pd.read_csv(file_path)
#
#     # 去除列名中的多余空格
#     data.columns = data.columns.str.strip()
#
#     # 检查是否包含'OT'列
#     if 'OT' not in data.columns:
#         raise ValueError("'OT' 列未找到，请检查数据文件的列名")
#
#     print(data.columns)
#
#     print(data['OT'].head())  # 打印前几行，确认数据
#
#     # 将 ILI 列转换为小数（百分比转小数）
#     data['WEIGHTED_ILI'] = data['WEIGHTED_ILI'] / 100
#     data['UNWEIGHTED_ILI'] = data['UNWEIGHTED_ILI'] / 100
#
#     # 将 'date' 列转换为日期格式
#     data['date'] = pd.to_datetime(data['date'])
#
#     # 按日期排序数据
#     data = data.sort_values(by='date')
#
#     print(data.head())  # 打印前几行，确认数据
#
#     return data
#
#
# # 创建数据集：特征和目标变量
# def create_dataset(data, features, target, time_steps=1):
#     X, y = [], []
#
#     # 打印列名，确认目标列是否存在
#     print("Columns in the data:", data.columns)
#
#     # 确保目标列 'OT' 存在
#     if target not in data.columns:
#         raise ValueError(f"目标列 '{target}' 未找到，请检查数据集列名")
#
#     for i in range(len(data) - time_steps):
#         X.append(data[features].iloc[i:i + time_steps].values)
#         y.append(data[target].iloc[i + time_steps])  # 目标列 'OT' 的预测值
#
#     return np.array(X), np.array(y)
#
#
# # 构建LSTM模型
# def build_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model
#
#
# # 主函数
# def main(file_path):
#     # 预处理数据
#     data = preprocess_data(file_path)
#
#     # 选择特征和目标列
#     features = ['WEIGHTED_ILI', 'UNWEIGHTED_ILI', 'AGE_0-4', 'AGE_5-24', 'ILITOTAL', 'NUM._OF_PROVIDERS','OT']
#     target = 'OT'
#
#     # 标准化特征数据
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data[features])
#
#     # 设置时间步长（例如：使用前10周的数据来预测下一个周的OT）
#     time_steps = 30
#     X, y = create_dataset(pd.DataFrame(scaled_data, columns=features), features, target, time_steps)
#
#     # 切分数据集为训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#     # 构建LSTM模型
#     model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#
#     # 训练模型
#     model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
#
#     # 预测结果
#     y_pred = model.predict(X_test)
#
#     # 评估模型
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"均方误差 (MSE): {mse}")
#
#     # 可视化真实值与预测值
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_test, color='blue', label='真实OT值')
#     plt.plot(y_pred, color='red', label='预测OT值')
#     plt.title('OT值的预测结果')
#     plt.xlabel('时间')
#     plt.ylabel('OT')
#     plt.legend()
#     plt.show()
#
#
# # 执行主函数
# file_path = '../dataset/illness/national_illness.csv'  # 数据文件路径
# main(file_path)


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# 设置字体为支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 如果 SimHei 没有，尝试 'Microsoft YaHei'

# 预处理函数：清理列名、标准化数据、准备特征和目标变量
def preprocess_data(file_path):
    # 读取数据
    data = pd.read_csv(file_path)

    # 去除列名中的多余空格
    data.columns = data.columns.str.strip()

    # 检查是否包含'OT'列
    if 'OT' not in data.columns:
        raise ValueError("'OT' 列未找到，请检查数据文件的列名")

    print(data.columns)

    print(data['OT'].head())  # 打印前几行，确认数据

    # 将 ILI 列转换为小数（百分比转小数）
    data['WEIGHTED_ILI'] = data['WEIGHTED_ILI'] / 100
    data['UNWEIGHTED_ILI'] = data['UNWEIGHTED_ILI'] / 100

    # 将 'date' 列转换为日期格式
    data['date'] = pd.to_datetime(data['date'])

    # 按日期排序数据
    data = data.sort_values(by='date')

    print(data.head())  # 打印前几行，确认数据

    return data


# 创建数据集：特征和目标变量
def create_dataset(data, features, target, time_steps=1):
    X, y = [], []

    # 打印列名，确认目标列是否存在
    print("Columns in the data:", data.columns)

    # 确保目标列 'OT' 存在
    if target not in data.columns:
        raise ValueError(f"目标列 '{target}' 未找到，请检查数据集列名")

    for i in range(len(data) - time_steps):
        X.append(data[features].iloc[i:i + time_steps].values)
        y.append(data[target].iloc[i + time_steps])  # 目标列 'OT' 的预测值

    return np.array(X), np.array(y)


# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 主函数
def main(file_path):
    # 预处理数据
    data = preprocess_data(file_path)

    # 选择特征和目标列
    features = ['WEIGHTED_ILI', 'UNWEIGHTED_ILI', 'AGE_0-4', 'AGE_5-24', 'ILITOTAL', 'NUM._OF_PROVIDERS','OT']
    target = 'OT'

    # 标准化特征数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    # 设置时间步长（例如：使用前30周的数据来预测下一个周的OT）
    time_steps = 100
    X, y = create_dataset(pd.DataFrame(scaled_data, columns=features), features, target, time_steps)

    # 切分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 构建LSTM模型
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # 训练模型
    start_time = time.time()
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    training_time = time.time() - start_time

    print(f"模型训练时间: {training_time:.2f}秒")

    # 预测结果
    y_pred = model.predict(X_test)

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    print(f"均方误差 (MSE): {mse}")

    # 可视化真实值与预测值
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='真实OT值')
    plt.plot(y_pred, color='red', label='预测OT值')
    plt.title('OT值的预测结果')
    plt.xlabel('时间')
    plt.ylabel('OT')
    plt.legend()
    plt.show()


# 执行主函数
file_path = '../dataset/illness/national_illness.csv'  # 数据文件路径
main(file_path)

