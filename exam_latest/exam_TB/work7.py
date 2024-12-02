import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# 设置字体为支持中文的字体
plt.rcParams['font.family'] = 'SimHei'

# 1. 从CSV文件加载数据
def load_data(filepath, column):
    """
    从CSV文件加载指定列的时间序列数据，并处理时间变化日。
    :param filepath: CSV文件路径
    :param column: 要预测的列（用户编号，例如 '0', '1'）
    :return: 选定列的时间序列数据（numpy数组）以及对应的时间索引
    """
    # 加载数据
    data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')

    # 处理10月时间增加日：合并凌晨1:00到2:00的值
    for year in range(2016, 2020):
        time_change_day = f"{year}-10-30 01:00:00"  # 时间变化日
        if time_change_day in data.index:
            # 将 1:00 和 2:00 的值合并到 1:00
            data.loc[time_change_day, column] += data.loc[f"{year}-10-30 02:00:00", column]
            data = data.drop(f"{year}-10-30 02:00:00")  # 删除重复的2:00值

    # 返回指定列的值和时间索引
    return data[column].values.reshape(-1, 1), data.index

# 2. 数据预处理
def preprocess_data(data, time_step):
    """
    归一化数据并生成时间序列数据。
    :param data: 原始数据
    :param time_step: 时间步长
    :return: 输入数据X、目标数据y和归一化器
    """
    # 使用 MinMaxScaler 将数据归一化到 [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # 创建时间序列样本
    X, y = [], []
    for i in range(len(data_scaled) - time_step):
        X.append(data_scaled[i:i + time_step, 0])  # 输入：前time_step步的数据
        y.append(data_scaled[i + time_step, 0])  # 输出：第time_step+1步的数据

    # 转换为 numpy 数组并调整为 LSTM 输入格式
    X = np.array(X).reshape(-1, time_step, 1)
    y = np.array(y)
    return X, y, scaler

# 3. 按时间划分训练集和测试集
def split_data_by_time(X, y, split_ratio=0.8):
    """
    根据时间顺序划分数据集。
    :param X: 输入数据
    :param y: 目标数据
    :param split_ratio: 训练集占比
    :return: 训练集和测试集
    """
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

# 4. 构建LSTM模型
def build_lstm_model(input_shape):
    """
    构建LSTM模型。
    :param input_shape: 输入数据的形状
    :return: 编译后的模型
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))  # LSTM层
    model.add(Dense(units=1))  # 全连接输出层
    model.compile(optimizer='adam', loss='mean_squared_error')  # 编译模型
    return model

# 5. 可视化结果
def plot_results(actual, predicted, dates, user_column):
    """
    可视化实际值与预测值（按小时显示）。
    :param actual: 实际值
    :param predicted: 预测值
    :param dates: 预测日期
    :param user_column: 用户编号
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, color='blue', label=f'实际值 (用户 {user_column})')
    plt.plot(dates, predicted, color='red', label=f'预测值 (用户 {user_column})')
    plt.title(f'用户 {user_column} 的用电量预测（未来 48 小时）')
    plt.xlabel('时间')
    plt.ylabel('用电量 (kW)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# 主函数：组合整个流程
def main(filepath, user_column='0', time_step=30, split_ratio=0.8, epochs=20, batch_size=32):
    """
    主函数，完成从数据加载到模型预测的完整流程。
    :param filepath: CSV文件路径
    :param user_column: 要预测的列（如 '0', '1', '2'）
    :param time_step: 时间步长
    :param split_ratio: 训练集占比
    :param epochs: 训练轮数
    :param batch_size: 批量大小
    """
    # 加载数据
    print(f"正在加载用户 {user_column} 的数据...")
    data, dates = load_data(filepath, column=user_column)

    # 数据预处理
    print("正在进行数据预处理...")
    X, y, scaler = preprocess_data(data, time_step)

    # 按时间划分训练集和测试集
    print("正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = split_data_by_time(X, y, split_ratio=split_ratio)

    # 构建模型
    print("正在构建LSTM模型...")
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))

    # 模型训练
    print("正在训练模型...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # 模型预测
    print("正在进行预测...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # 反归一化预测值
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # 反归一化实际值

    # 预测未来48小时的用电量
    future_predictions = predictions[-48:]  # 预测最后48小时的数据
    future_actual = y_test_actual[-48:]  # 获取实际的最后48小时的数据

    # 获取未来48小时的时间标签（日期）
    future_dates = dates[-48:]

    # 可视化结果
    print("绘制结果...")
    plot_results(future_actual, future_predictions, future_dates, user_column)


# 运行主程序
if __name__ == "__main__":
    filepath = "../dataset/electricity/electricity.csv"  # 替换为您的CSV文件路径
    user_column = input("请输入要预测的用户编号 (例如 '0', '1'): ")
    main(filepath, user_column=user_column)
