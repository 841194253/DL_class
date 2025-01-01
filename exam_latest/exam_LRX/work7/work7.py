# -*- coding: utf-8 -*-

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


import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import TimeDistributed, RepeatVector, ConvLSTM2D
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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


# 构建不同的LSTM模型
def build_lstm_model(input_shape, n_outputs=1):
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=False, input_shape=input_shape))
    model.add(Dense(n_outputs))  # 输出层的神经元数目根据n_outputs决定
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape'])
    print(model.summary())
    return model


def build_advanced_lstm_model(input_shape, n_outputs=1):
    model = Sequential()

    model.add(Bidirectional(LSTM(units=64, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(n_outputs))  # 输出层根据n_outputs动态调整
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae', 'mape'])
    model.build(input_shape=(None, input_shape[0], input_shape[1]))
    print(model.summary())
    return model


def build_conv1d_lstm_model(input_shape, n_outputs=1):
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(0.3))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(n_outputs))  # 输出层根据n_outputs动态调整

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae', 'mape'])
    print(model.summary())

    return model


def conv_lstm_ConvLSTM2D_model(input_shape, n_outputs=1):
    """
    构建一个基于ConvLSTM的神经网络模型，适用于时间序列预测任务。

    :param input_shape: 输入数据的形状，(时间步数, 特征维度, 通道数)
    :param n_outputs: 输出的时间步数

    :return: 编译后的模型
    """
    # 构建模型
    model = Sequential()

    # ConvLSTM2D层：处理时间序列数据并提取时空特征
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                         input_shape=input_shape))

    # Flatten层：将多维输出展平为一维
    model.add(Flatten())

    # RepeatVector层：将输入重复n_outputs次，以便输出多个时间步
    model.add(RepeatVector(n_outputs))

    # LSTM层：学习时间序列的动态特征
    model.add(LSTM(200, activation='relu', return_sequences=True))

    # TimeDistributed层：应用Dense层到每一个时间步
    model.add(TimeDistributed(Dense(100, activation='relu')))

    # TimeDistributed层：输出每个时间步的预测
    model.add(TimeDistributed(Dense(1)))

    # 编译模型
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])

    # 打印模型概况
    print(model.summary())

    return model

# 选择并构建模型
def select_model(model_type, input_shape, n_outputs=1):
    if model_type == 'lstm':
        return build_lstm_model(input_shape, n_outputs)
    elif model_type == 'advanced_lstm':
        return build_advanced_lstm_model(input_shape, n_outputs)
    elif model_type == 'conv1d_lstm':
        return build_conv1d_lstm_model(input_shape, n_outputs)
    elif model_type == 'conv2d_lstm':
        return conv_lstm_ConvLSTM2D_model(input_shape, n_outputs)
    else:
        raise ValueError(f"未识别的模型类型: {model_type}")


# 数据可视化（展示每个特征随时间的变化）
def show_raw_visualization(data, feature_keys):
    """
    数据可视化
    :param data: 数据字典，包含特征和时间戳数据
    :return: None
    """
    time_data = data['date']
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15), dpi=80, facecolor="w", edgecolor="k")

    titles = [
        "WEIGHTED ILI",
        "UNWEIGHTED ILI",
        "AGE 0-4",
        "AGE 5-24",
        "ILITOTAL",
        "NUM. OF PROVIDERS",
        "OT"
    ]

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]

    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % len(colors)]
        t_data = data[key]
        t_data.index = time_data

        ax = t_data.plot(ax=axes[i // 2, i % 2], color=c, title=f"{titles[i]} - {key}", rot=25)
        ax.legend([titles[i]])

        # 设置y轴格式，防止科学计数法
        ax.yaxis.set_major_formatter(ScalarFormatter())  # 使用普通格式显示
        ax.yaxis.get_major_formatter().set_scientific(False)  # 禁止使用科学计数法

    plt.tight_layout()
    plt.savefig('data.png')
    plt.show()


# 主函数
def main(file_path, model_type='lstm'):
    # 预处理数据
    data = preprocess_data(file_path)

    data['date'] = pd.to_datetime(data['date'])

    # 选择特征和目标列
    features = ['WEIGHTED_ILI', 'UNWEIGHTED_ILI', 'AGE_0-4', 'AGE_5-24', 'ILITOTAL', 'NUM._OF_PROVIDERS', 'OT']
    target = 'OT'

    show_raw_visualization(data, features)

    # 标准化特征数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    # 获取目标列的scaler，用于反归一化
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_data = data[[target]]
    target_scaler.fit(target_data)

    # 设置时间步长（例如：使用前time_steps周的数据来预测下一个周的OT）
    time_steps = 200
    X, y = create_dataset(pd.DataFrame(scaled_data, columns=features), features, target, time_steps)

    # 切分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 动态选择并构建模型
    model = select_model(model_type=model_type, input_shape=(X_train.shape[1], X_train.shape[2]), n_outputs=1)

    # 定义学习率衰减
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # 训练模型
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping], verbose=1)
    # history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),callbacks=[reduce_lr], verbose=1)
    training_time = time.time() - start_time
    print(f"模型训练时间: {training_time:.2f}秒")

    # 绘制损失曲线
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title("训练与验证损失")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

    # 预测结果
    y_pred = model.predict(X_test)

    # 反归一化结果
    y_pred_rescaled = target_scaler.inverse_transform(y_pred)
    y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))


    # 计算MSE和MAE
    mse = mean_squared_error(y_test, y_pred)  # y_test 和 y_pred 是归一化后的数据
    mae = mean_absolute_error(y_test, y_pred)

    print(f"MSE: {mse:.7f}")
    print(f"MAE: {mae:.7f}")

    # 创建一个新的 DataFrame 用于保存预测结果
    prediction_df = pd.DataFrame({
        'y_test_rescaled': y_test_rescaled.flatten(),
        'y_pred_rescaled': y_pred_rescaled.flatten()
    })

    # 提取测试集日期
    test_dates = data['date'].iloc[-len(y_test_rescaled):]

    # 绘制预测结果
    plt.plot(test_dates, y_test_rescaled, label='真实值', color='blue', linewidth=2)
    plt.plot(test_dates, y_pred_rescaled, label='预测值', color='red',  linewidth=2)

    # 设置y轴格式，去除科学计数法
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())  # 使用普通格式显示
    plt.gca().yaxis.get_major_formatter().set_scientific(False)  # 禁止使用科学计数法
    plt.legend()

    plt.grid(True)  # 添加网格线，增加可读性
    # 自动旋转标签，避免重叠
    plt.gcf().autofmt_xdate()
    plt.title('真实值与预测值')
    plt.xlabel('时间')
    plt.ylabel('患者数量')
    plt.savefig('predictions_true.png')
    plt.show()


if __name__ == '__main__':
    file_path = '../../dataset/illness/national_illness.csv'
    model_type = 'lstm'  # 可以选择 'lstm', 'advanced_lstm', 或 'conv1d_lstm' conv2d_lstm
    main(file_path, model_type=model_type)

# 模型优缺点分析：
# 优点：
# 长短期记忆： LSTM（长短期记忆网络）能够处理和预测时间序列数据中的长期依赖关系，克服了传统 RNN（递归神经网络）在处理长序列时容易出现梯度消失或梯度爆炸的问题。
# 自动学习特征： LSTM 网络可以自动从数据中学习特征，无需人工选择或设计特征。
# 适应复杂时间序列： LSTM 网络特别适用于处理具有复杂模式的时间序列数据，能够捕捉到数据中的周期性、趋势性以及突变等多种变化规律。
# 强大的记忆能力： LSTM 可以通过门控机制决定哪些信息需要记忆，哪些信息可以遗忘，这使得其在处理需要“记住”长时间信息的任务中表现良好。
# 缺点：
# 计算资源消耗： LSTM 结构复杂，训练时间较长，尤其是在大量数据和深层网络结构时，计算资源消耗较大。
# 过拟合风险： 如果训练数据过少，或者模型参数过多，LSTM 容易出现过拟合，导致模型在测试集上的泛化能力较差。
# 超参数调节： LSTM 模型包含大量超参数（如学习率、隐藏层单元数量、批次大小等），调节这些参数需要时间和经验。
# 需要大规模数据： LSTM 网络通常需要大量的数据进行训练，才能避免过拟合和提高模型的预测能力。

# 数据特性（如周期性、趋势性）对模型表现的影响
# 周期性：
# 时间序列数据中常见的周期性特征（例如季节性变化、日夜交替等）对 LSTM 模型的表现有重要影响。如果数据具有强周期性，LSTM 可以通过其门控机制学到周期性模式并有效进行预测。尤其是 LSTM 能够记住周期性变化并根据时间序列的历史信息来预测未来的周期波动。因此，对于周期性强的时间序列（例如气温、股票的波动等），LSTM 会比简单的线性模型或其他传统方法表现得更好。
# 趋势性：
# 趋势性数据指的是随着时间推移，数据呈现出一定的上升或下降趋势。LSTM 模型能够学习并捕捉这种趋势性，尤其是在长时间序列数据中，LSTM 能够通过其记忆功能从过去的长期趋势中预测未来的变化。如果数据中的趋势性较强，LSTM 可以在更长的时间跨度上保持较好的预测效果。然而，过于复杂的趋势性可能会导致 LSTM 模型的训练更加困难，因此合理设计模型架构和选择优化算法是非常关键的。
# 噪声和突变：
# LSTM 也能处理具有一些噪声或突变的时间序列数据。虽然 LSTM 的记忆机制可以捕捉数据中的异常点，但过多的噪声会影响模型的稳定性和预测精度。为了有效处理噪声数据，可以考虑在数据预处理阶段使用滤波、去噪等技术来增强模型的鲁棒性。
# LSTM 原理和优势：
# LSTM 是一种特殊的 RNN，它引入了三种门（遗忘门、输入门、输出门），这些门控机制使得 LSTM 能够在长时间的序列中保持长期的依赖关系。
# 遗忘门（Forget Gate）： 决定了哪些信息会从记忆单元中丢失。
# 输入门（Input Gate）： 控制新的信息如何进入记忆单元。
# 输出门（Output Gate）： 确定当前记忆的输出信息。
# 这些门的存在使得 LSTM 能够通过“记住”重要的信息并“忘记”无关的部分来捕捉长期的时间依赖性，这对于处理具有长短期依赖关系的时间序列数据至关重要。
# LSTM 在时间序列中的优势：
# 记住长期信息： LSTM 的门控机制使其能够在较长时间段内保持信息，从而能够准确地捕捉到时间序列中的长期依赖。
# 灵活的学习能力： LSTM 在学习过程中不仅关注当前时刻的信息，也可以通过历史时间步的信息来进行预测，这种灵活性使其能够适应各种复杂的时间序列特征。
# 避免梯度消失问题： 传统的 RNN 在长序列的训练过程中容易出现梯度消失问题，但 LSTM 通过其独特的结构（如遗忘门）有效地避免了这一问题，使其在长时间序列预测任务中表现更好。

# 人数的预测已有图表显示 趋势是对的 但是数据不准