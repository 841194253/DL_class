import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import Sequential

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


# 优点：
# - 模型精度高：LSTM模型在处理时间序列问题时特别有效，能捕捉数据中的长期依赖性，能够根据历史数据预测未来汇率变化趋势。
# - 早停机制：通过`EarlyStopping`回调函数，避免了过拟合现象，可以提高模型在验证集上的表现，同时通过恢复最优权重确保训练过程稳定。
#
# 缺点：
# - 计算资源消耗大：LSTM模型通常需要较长的训练时间，并且对硬件要求较高，尤其是数据量较大时，训练时间会显著增加。
# - 对超参数敏感：LSTM对一些超参数（如`look_back`、LSTM层单元数、学习率等）非常敏感，可能需要通过网格搜索或其他方法调优来获得更好的表现。
# - 对周期性和趋势性表现不如经典模型：LSTM在捕捉长期依赖性方面表现优秀，但对于一些具有明确周期性和趋势性的时间序列（如汇率），可能并不是最优选择。经典的ARIMA、SARIMA等模型可能在某些场景下表现更好。
#
# 改进方向：
# - 超参数优化：可以进一步优化`look_back`、LSTM单元数、批次大小等超参数。
# - 加入外部特征：如经济数据、政策变动等，结合多元时间序列分析，提高预测准确度。
# - 增加模型复杂度：通过增加层数或采用Bidirectional LSTM、GRU等变种，可能提高模型在复杂时间序列中的表现。
# - 集成学习：通过结合多个LSTM模型或与其他模型（如XGBoost）集成，可能进一步提升预测效果。
#
# 2. 讨论数据特性（如周期性、趋势性）对模型表现的影响：
#
# 周期性：
# - 汇率数据可能受到季节性因素、节假日、贸易周期等因素的影响。如果模型未能有效捕捉到这些周期性特征，可能会导致预测效果不理想。LSTM模型虽然能够捕捉长短期依赖，但可能无法直接学习到周期性变化。
# - 改进方法：可以通过将周期性特征作为额外的输入特征（如月份、季度等）来增强模型对周期性变化的学习能力。
#  趋势性：
# - 如果数据存在明显的趋势性（如汇率逐年上涨或下跌），LSTM可能需要较长时间才能学习到趋势，尤其在数据量较小时。
# - 改进方法：可以使用趋势分解方法，如Holt-Winters法或其他趋势平滑方法，先对数据进行预处理，去除趋势成分，再进行建模。
# 对模型表现的影响：
# - LSTM优势：LSTM能够自动学习并捕捉数据中的长期依赖性，这使得它特别适合捕捉汇率数据中的短期波动，但它对数据的长期趋势和周期性变化的适应性较差。
# - 其他模型优势：经典时间序列模型（如ARIMA）通常能够较好地处理趋势性和周期性强的数据，能够更准确地把握数据的整体变化趋势。
#  3.阐述LSTM的原理，特别是其在处理时间序列数据中的长短期记忆优势：**
#  LSTM（Long Short-Term Memory）：
# LSTM是一种特殊的RNN（循环神经网络），通过引入**遗忘门**、**输入门**和**输出门**，使得模型在处理序列数据时，能够有效保留长期依赖关系，并避免传统RNN面临的梯度消失和爆炸问题。其结构如下：
# - 遗忘门（Forget Gate）：决定了哪些信息需要丢弃。它根据当前输入和之前的隐状态生成一个0到1之间的数字，表示每个单元信息的保留程度。
# - 输入门（Input Gate）：决定了哪些信息需要加入到单元状态中，包含对当前输入和上一个状态的加权计算。
# - 输出门（Output Gate）：决定了当前单元的输出，基于输入门的控制，以及上一时刻的单元状态。
# 在处理时间序列中的优势：
# - 长期记忆：LSTM能够捕捉长时间跨度内的数据依赖，使得它能够利用历史数据预测未来趋势，适用于预测汇率、股市等具有长期依赖性的时间序列数据。
# - 短期记忆：LSTM可以通过遗忘门丢弃无关信息，这使得它能更加专注于当前和未来时刻的相关数据，从而提高了短期预测的精度。
# - 非线性处理：相比传统的ARIMA模型，LSTM能够处理数据中的非线性关系，这对于汇率这种受多种因素影响的时间序列数据来说尤为重要。
#  在汇率预测中的应用：
# - 短期预测：LSTM能够学习历史汇率数据中的模式，从而进行短期内的准确预测。
# - 长期趋势：尽管LSTM擅长处理长期依赖，但在一些具有强趋势性的数据中，可能依然需要结合其他方法来更好地捕捉长期趋势。

# 分析结果已经保存 每个国家的汇率预测真实值和预测值都已经作出图表 结果接近但是很依旧有差距 并且可以调整look_back来增加训练数据