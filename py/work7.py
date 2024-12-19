import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from keras import layers
from keras import models

# 设置字体为支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 如果 SimHei 没有，尝试 'Microsoft YaHei'

# 载入数据集（假设数据已经保存在本地 CSV 文件）
df = pd.read_csv("../exam_latest/dataset/illness/national_illness.csv")

# 确保 'date' 列是按周颗粒度的时间戳，并将其转为 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 确保按时间顺序排列
df = df.sort_values(by=['date'])

# 选择特征列
feature_keys = [
    "WEIGHTED_ILI",
    "UNWEIGHTED_ILI",
    "AGE_0-4",
    "AGE_5-24",
    "ILITOTAL",
    "NUM._OF_PROVIDERS",
    "OT"
]

# 选择特征并标准化
features = df[feature_keys]

# 数据归一化
scaler = sklearn.preprocessing.StandardScaler()
features_scaled = scaler.fit_transform(features)

# 将标准化后的数据转换回 DataFrame
features_scaled = pd.DataFrame(features_scaled, columns=feature_keys)

# 按时间划分训练集和验证集（按时间顺序划分）
train_split = int(0.8 * len(df))  # 80% 训练集，20% 验证集
train_data = features_scaled[:train_split]
val_data = features_scaled[train_split:]

# 设置序列长度和步长
past = 720  # 模型用过去720个时间步的数据
future = 72  # 预测未来72个时间步的数据
step = 6  # 每6个时间步取一次数据
sequence_length = int(past / step)


# 创建数据集（输入输出数据对）
def create_dataset(data, sequence_length, step):
    X, y = [], []
    for i in range(len(data) - sequence_length - future):
        X.append(data.iloc[i:i + sequence_length, :-1].values)  # 选择除目标列外的所有列作为输入特征
        y.append(data.iloc[i + sequence_length + future - 1, -1])  # 目标列即OT
    return np.array(X), np.array(y)


# 创建训练和验证数据集
X_train, y_train = create_dataset(train_data, sequence_length, step)
X_val, y_val = create_dataset(val_data, sequence_length, step)

# 定义LSTM模型
# 使用 Sequential 构建模型
model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))  # 使用 LSTM 层时定义输入形状
model.add(layers.Dense(1))  # 添加输出层

model.compile(optimizer='adam', loss='mse')
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1)


# 绘制训练过程中的损失值
def visualize_loss(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='训练损失')
    plt.plot(epochs, val_loss, 'r', label='验证损失')
    plt.title("训练和验证损失")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# 绘制损失曲线
visualize_loss(history)

# 使用训练好的模型进行预测
predictions = model.predict(X_val)

# 将预测值与实际值一起展示
plt.figure(figsize=(10, 6))
plt.plot(y_val, label='真实值')
plt.plot(predictions, label='预测值')
plt.title("预测与真实值对比")
plt.xlabel("时间步")
plt.ylabel("OT")
plt.legend()
plt.show()


# 数据可视化（展示每个特征随时间的变化）
def show_raw_visualization(data):
    """
    数据可视化
    :param data: 数据字典，包含特征和时间戳数据
    :param date_time_key: 时间数据的键
    :param feature_keys: 要可视化的特征的键列表
    :param titles: 对应特征数据的标题列表
    :param colors: 为每个特征指定的颜色列表
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

    plt.tight_layout()
    plt.show()


# 展示原始数据可视化
show_raw_visualization(df)
