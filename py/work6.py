# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# from keras._tf_keras.keras.utils import to_categorical
# from keras._tf_keras.keras.models import Sequential
# from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, BatchNormalization
# from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Input
# from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
# from keras._tf_keras.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt
#
# # 设置 KERAS_BACKEND 环境变量
# os.environ['KERAS_BACKEND'] = 'tensorflow'
#
#
# def load_and_preprocess_cifar10(data_dir):
#     """
#     从本地加载并预处理 CIFAR-10 数据集。
#     参数：
#         data_dir: 本地 CIFAR-10 数据集目录。
#     返回：
#         训练集和测试集数据及其标签。
#     """
#     x_train = []
#     y_train = []
#     x_test = []
#     y_test = []
#
#     # 加载训练集
#     for i in range(1, 6):
#         with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
#             batch = pickle.load(f, encoding='latin1')
#             x_train.append(batch['data'])
#             y_train += batch['labels']
#
#     # 加载测试集
#     with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
#         test_batch = pickle.load(f, encoding='latin1')
#         x_test.append(test_batch['data'])
#         y_test = test_batch['labels']
#
#     # 数据形状转换 (N, H, W, C)
#     x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#     x_test = np.concatenate(x_test).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#
#     # 数据归一化
#     x_train = x_train.astype('float32') / 255.0
#     x_test = x_test.astype('float32') / 255.0
#
#     # One-Hot 编码
#     y_train = to_categorical(np.array(y_train), 10)
#     y_test = to_categorical(np.array(y_test), 10)
#
#     return (x_train, y_train), (x_test, y_test)
#
#
# def plot_cifar10_images(images, labels, num_images=10):
#     """
#     可视化 CIFAR-10 图像及其标签。
#     参数：
#         images: 图像数据 (numpy 数组)。
#         labels: 真实标签 (numpy 数组)。
#         num_images: 显示的图像数量。
#     """
#     plt.figure(figsize=(12, 12))
#     for i in range(num_images):
#         plt.subplot(5, 5, i + 1)
#         plt.imshow(images[i])
#         plt.title(label_dict[np.argmax(labels[i])])
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()
#
#
# def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
#     """
#     创建卷积神经网络模型。
#     参数：
#         input_shape: 输入数据的形状。
#         num_classes: 分类类别数。
#     返回：
#         创建的 CNN 模型。
#     """
#     model = Sequential([
#         Input(shape=input_shape),  # 显式添加输入层
#         Conv2D(32, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Dropout(0.25),
#         MaxPooling2D((2, 2)),
#
#         Conv2D(64, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Dropout(0.25),
#         MaxPooling2D((2, 2)),
#
#         Conv2D(128, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Dropout(0.25),
#         MaxPooling2D((2, 2)),
#
#         Flatten(),
#         Dense(512, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])
#     return model
#
#
# def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
#     """
#     训练模型。
#     参数：
#         model: CNN 模型。
#         x_train: 训练集特征。
#         y_train: 训练集标签。
#         x_val: 验证集特征。
#         y_val: 验证集标签。
#         epochs: 训练轮数。
#         batch_size: 批量大小。
#     返回：
#         训练历史记录。
#     """
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
#     datagen.fit(x_train)
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
#     train_history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
#                               validation_data=(x_val, y_val),
#                               epochs=epochs,
#                               callbacks=[early_stopping],
#                               verbose=2)
#
#     return train_history
#
#
# def show_train_history(train_history):
#     """
#     显示训练历史。
#     参数：
#         train_history: 训练历史记录。
#     """
#     plt.plot(train_history.history['accuracy'])
#     plt.plot(train_history.history['val_accuracy'])
#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()
#
#     plt.plot(train_history.history['loss'])
#     plt.plot(train_history.history['val_loss'])
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()
#
#
# def main(data_dir):
#     """
#     主函数。
#     参数：
#         data_dir: CIFAR-10 数据集本地路径。
#     """
#     if not all(os.path.exists(os.path.join(data_dir, f'data_batch_{i}')) for i in range(1, 6)) or \
#        not os.path.exists(os.path.join(data_dir, 'test_batch')):
#         print("Dataset not found.")
#         return
#
#     (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10(data_dir)
#
#     # 划分验证集
#     x_val = x_train[:10000]
#     y_val = y_train[:10000]
#     x_train = x_train[10000:]
#     y_train = y_train[10000:]
#
#     # 可视化部分训练集图片
#     plot_cifar10_images(x_train, y_train, num_images=10)
#
#     # 创建和编译模型
#     model = create_cnn_model()
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     # 训练模型
#     train_history = train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=128)
#
#     # 测试模型性能
#     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
#     print(f"Test Accuracy: {test_acc * 100:.2f}%")
#
#     # 显示训练历史
#     show_train_history(train_history)
#
#
# # 类别标签字典
# label_dict = {
#     0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
#     4: "deer", 5: "dog", 6: "frog", 7: "horse",
#     8: "ship", 9: "truck"
# }
#
#
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#
# # 设置本地数据集路径并运行主程序
# data_dir = '../images/CIFAR-10'  # 替换为实际数据集路径
# main(data_dir)

# import os
# os.environ['KERAS_BACKEND'] = 'torch'
# import keras
# print(keras.__version__)
#
# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())

# import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'

# import numpy as np
# import os
#
# # 导入keras之前设置KERAS_BACKEND环境变量
# os.environ['KERAS_BACKEND'] = 'tensorflow'
#
# import keras
#
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
#
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
#
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
#
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")
# print(keras.__version__)
# print("Backend:", keras.config.backend())


# import os
#
# # Set backend env to tensorflow
# os.environ["KERAS_BACKEND"] = "tensorflow"

import os
import pickle

# from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator
# from keras import preprocessing 因为keras3的preprocessing没了ImageDataGenerator
# from keras.callbacks import EarlyStopping
from keras import callbacks
# from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras import layers
# from keras.utils import to_categorical
from keras import utils

os.environ['KERAS_BACKEND'] = 'tensorflow'


def load_and_preprocess_cifar10(data_dir):
    """加载并预处理 CIFAR-10 数据集。"""
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # 读取训练集
    for i in range(1, 6):
        with open(os.path.join(data_dir, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="latin1")
            x_train.append(batch["data"])
            y_train += batch["labels"]

    # 读取测试集
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        test_batch = pickle.load(f, encoding="latin1")
        x_test.append(test_batch["data"])
        y_test = test_batch["labels"]

    # 数据转换与重塑
    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = np.concatenate(x_test).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """创建卷积神经网络模型。"""
    model = Sequential([
        layers.Input(shape=input_shape),  # 使用 Input 层代替在 Conv2D 层中指定 input_shape

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def prepare_tf_dataset(x, y, batch_size=128, shuffle=True):
    """将数据封装为 TensorFlow Dataset 并进行增强。"""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))  # 随机打乱数据
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # 异步预取数据以提高性能
    return dataset


def train_model(model, train_ds, val_ds, epochs=50):
    """训练模型，使用早期停止防止过拟合。"""
    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # 开始训练
    train_history = model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=[early_stopping],
                              verbose=2)

    return train_history


def plot_images(images, labels, predictions=None, num_images=10):
    """可视化图像与标签。"""
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        true_label = np.argmax(labels[i])
        title = f"True: {label_dict[true_label]}"
        if predictions is not None:
            pred_label = np.argmax(predictions[i])
            title += f" | Pred: {label_dict[pred_label]}"

        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """可视化训练过程中的损失和精度。"""
    # 获取训练和验证的损失和精度
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(train_loss) + 1)  # 获取训练轮数

    # 绘制训练和验证损失图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 1行2列，第1个图
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和验证精度图
    plt.subplot(1, 2, 2)  # 1行2列，第2个图
    plt.plot(epochs, train_acc, 'b', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()


def main(data_dir):
    """主函数，执行数据集的加载、模型创建、训练和预测。"""
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10(data_dir)
    x_val, y_val = x_train[:10000], y_train[:10000]
    x_train, y_train = x_train[10000:], y_train[10000:]

    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

    # 使用 tf.data.Dataset
    train_ds = prepare_tf_dataset(x_train, y_train)
    val_ds = prepare_tf_dataset(x_val, y_val, shuffle=False)

    model = create_cnn_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())

    # 训练模型并获取训练历史
    train_history = train_model(model, train_ds, val_ds)

    # 可视化训练过程中的精度与损失
    plot_training_history(train_history)

    # 在测试集上进行预测并展示图像
    predictions = model.predict(x_test)
    plot_images(x_test, y_test, predictions, num_images=10)


# 标签字典
label_dict = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse",
    8: "ship", 9: "truck"
}

data_dir = "../images/CIFAR-10"  # 数据集路径
main(data_dir)




