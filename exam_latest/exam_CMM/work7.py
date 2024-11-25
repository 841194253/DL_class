import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import math
import os
from matplotlib.ticker import MaxNLocator

# 设置字体为支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 如果 SimHei 没有，尝试 'Microsoft YaHei'


# 读取并处理数据
def read_and_preprocess(file_path, look_back=30):
    data = pd.read_csv(file_path, parse_dates=[0], index_col=0)
    dates = data.index
    countries_data = data.values

    X_dict, y_dict, used_data_dict = {}, {}, {}
    for column_index in range(countries_data.shape[1]):
        X, y, used_data = [], [], []
        for i in range(len(countries_data) - look_back):
            X.append(countries_data[i:i + look_back, column_index].reshape(-1, 1))
            y.append(countries_data[i + look_back, column_index])
            used_data.append(dates[i + look_back])
        X_dict[column_index], y_dict[column_index], used_data_dict[column_index] = np.array(X), np.array(y), np.array(
            used_data)

    return X_dict, y_dict, used_data_dict, dates, data


# 搭建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', recurrent_activation='sigmoid', input_shape=input_shape, unroll=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

import time

# 训练LSTM模型
def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, log_file="training_log.txt"):
    # 创建日志文件并记录标题
    with open(log_file, "w") as f:
        f.write("Epoch, Train Loss, Val Loss, Time per Epoch (s)\n")

    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集的损失
        patience=10,  # 允许验证损失没有改善的周期数
        min_delta=0.001,  # 最小的损失改善，只有变化大于此值时才认为有改善
        restore_best_weights=True  # 恢复最优权重
    )

    start_time = time.time()  # 记录开始时间
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    # 打印history的内容以确认是否正确保存了训练数据
    print("训练历史记录:", history.history)  # 输出history对象的内容以便调试

    # 训练过程详细输出
    for epoch in range(len(history.history['loss'])):  # 使用history.history的长度来循环
        epoch_time = time.time() - start_time  # 计算每个epoch的时间
        train_loss = history.history['loss'][epoch]  # 训练损失
        val_loss = history.history['val_loss'][epoch]  # 验证损失
        print(
            f"Epoch {epoch + 1}/{len(history.history['loss'])} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")

        # 将每个epoch的损失和时间记录到日志文件
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}, {epoch_time:.2f}\n")

    return history


# 使用测试集进行预测
def predict_lstm_model(model, X_test):
    predictions = model.predict(X_test)
    return predictions


from matplotlib.ticker import MaxNLocator

# 绘制训练过程中的损失和验证损失
def plot_training_history(history, country_name, ax):
    ax.plot(history.history['loss'], label=f'{country_name} 训练损失')
    ax.plot(history.history['val_loss'], label=f'{country_name} 验证损失')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title(f'{country_name} 训练损失与验证损失')

    # 设置y轴的精度为0.001
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both', steps=[1, 2, 5, 10]))  # 设置步长精度
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.6f}'))  # 设置显示的格式为保留三位小数


# 绘制预测结果与实际值对比图
def plot_predictions(actual_values, predicted_values, country_name, ax, label_actual='实际值', label_predicted='预测值'):
    ax.plot(actual_values, label=f'{country_name} {label_actual}', color='blue')
    ax.plot(predicted_values, label=f'{country_name} {label_predicted}', color='red')
    ax.set_xlabel('日期')
    ax.set_ylabel('汇率')
    ax.legend()
    ax.set_title(f'{country_name} 预测与实际值对比')

    # 设置y轴的精度为0.1
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both', steps=[1, 2, 5, 10]))  # 设置步长精度
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  # 设置显示的格式为保留一位小数



# 主程序
def main(file_path, save_path="output_images/"):
    look_back = 30
    X_dict, y_dict, used_data_dict, dates, data = read_and_preprocess(file_path, look_back)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for country_column_index in range(data.shape[1]):
        print(f"\n国家 {country_column_index} 训练集前两天数据：")
        for i in range(2):
            date = dates[i]
            value = data.iloc[i, country_column_index]
            print(f"训练集 - 日期: {date}, 汇率: {value}")

        train_size = int(len(X_dict[country_column_index]) * 0.8)
        print(f"国家 {country_column_index} 验证集前两天数据：")
        for i in range(2):
            date = dates[train_size + i]
            value = data.iloc[train_size + i, country_column_index]
            print(f"验证集 - 日期: {date}, 汇率: {value}")

        test_size = len(X_dict[country_column_index]) - train_size
        X_test = X_dict[country_column_index][train_size:]
        y_test = y_dict[country_column_index][train_size:]
        print(f"国家 {country_column_index} 测试集前两天数据：")
        for i in range(2):
            date = dates[train_size + test_size + i]
            value = data.iloc[train_size + test_size + i, country_column_index]
            print(f"测试集 - 日期: {date}, 汇率: {value}")

        X_train, X_val = X_dict[country_column_index][:train_size], X_dict[country_column_index][train_size:]
        y_train, y_val = y_dict[country_column_index][:train_size], y_dict[country_column_index][train_size:]

        model = build_lstm_model((X_train.shape[1], 1))

        history = train_lstm_model(model, X_train, y_train, X_val, y_val)

        predictions_val = predict_lstm_model(model, X_val)
        predictions_test = predict_lstm_model(model, X_test)

        print(f"国家 {country_column_index} 预测日期数据：")
        for i in range(2):
            date = dates[train_size + i]
            predicted_value = predictions_val[i][0]
            print(f"验证集预测日期 - 日期: {date}, 预测汇率: {predicted_value:.4f}")

        for i in range(2):
            date = dates[train_size + test_size + i]
            predicted_value = predictions_test[i][0]
            print(f"测试集预测日期 - 日期: {date}, 预测汇率: {predicted_value:.4f}")

        fig, axs = plt.subplots(3, 1, figsize=(10, 18))

        plot_training_history(history, f'国家 {country_column_index}', axs[0])

        plot_predictions(y_val, predictions_val, f'国家 {country_column_index}', axs[1], label_actual='验证集实际值',
                         label_predicted='验证集预测值')

        plot_predictions(y_test, predictions_test, f'国家 {country_column_index}', axs[2], label_actual='测试集实际值',
                         label_predicted='测试集预测值')

        plt.tight_layout()
        fig.savefig(f"{save_path}country_{country_column_index}_train_val_test_predict.png")
        plt.close(fig)


file_path = '../dataset/exchange_rate/exchange_rate.csv'
main(file_path, save_path="output_images/")
