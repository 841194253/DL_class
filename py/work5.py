# 使用小数据集（例如CIFAR-10或MNIST）实现简单的卷积神经网络（CNN）进行图像分类。讨论层的选择（例如卷积、池化）如何影响性能。

import os
import pickle
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def load_and_preprocess_cifar10(data_dir):
    """
        加载并预处理 CIFAR-10 数据集。

        参数：
        data_dir: 存储数据集的目录路径。

        返回：
        包含训练集和测试集的特征和标签。
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # 训练集文件
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            x_train.append(batch['data'])
            y_train += batch['labels']

    # 测试集文件
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        x_test.append(test_batch['data'])
        y_test = test_batch['labels']

    # 将数据拼接在一起
    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)
    y_train = np.array(y_train)
    x_test = np.concatenate(x_test).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)

    # 数据归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-Hot Encoding
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def plot_cifar10_images(images, labels, num_images=10):
    """
    可视化 CIFAR-10 图像及其标签

    参数：
    images: 图像数据 (numpy 数组)
    labels: 真实标签 (numpy 数组)
    num_images: 显示的图像数量
    """
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(label_dict[np.argmax(labels[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_images_labels_prediction(images, labels, predictions, start_index, num_images=10):
    """
    可视化图像及其真实标签和预测标签

    参数：
    images: 图像数据 (numpy 数组)
    labels: 真实标签 (numpy 数组)
    predictions: 预测标签 (numpy 数组)
    start_index: 从哪个索引开始显示图像
    num_images: 显示的图像数量，最多为 25 张
    """
    if images.max() > 1.0:
        images = images / 255.0
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    num_images = min(num_images, 25)

    for i in range(num_images):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[start_index + i])

        true_label = np.argmax(labels[start_index + i])
        pred_label = np.argmax(predictions[start_index + i]) if predictions is not None else "N/A"

        title = f"{i + 1}: {label_dict[true_label]} => {label_dict[pred_label]}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('predictions.png')
    plt.show()

def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """
        创建卷积神经网络模型。

        参数：
        input_shape: 输入数据的形状（高度, 宽度, 通道数）。
        num_classes: 输出类别的数量。

        返回：
        model:构建好的CNN模型。
    """
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
    """
        训练模型。

        参数：
        model:要训练的模型。
        x_train: numpy 数组，训练数据的特征。
        y_train: numpy 数组，训练数据的标签。
        x_val: numpy 数组，验证数据的特征。
        y_val: numpy 数组，验证数据的标签。
        epochs:训练的轮数。
        batch_size:批次大小。

        返回：
        train_history:训练过程中的历史记录。
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    train_history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                              validation_data=(x_val, y_val),
                              epochs=epochs,
                              callbacks=[early_stopping],
                              verbose=2)

    return train_history

def show_train_history(train_acc, test_acc,train_history):
    """
    显示训练历史，包括训练和测试的准确率。

    参数：
    train_acc: 训练准确率历史数据。
    test_acc: 测试准确率历史数据。
    """
    plt.plot(train_history.history[train_acc])  # 绘制训练准确率
    plt.plot(train_history.history[test_acc])  # 绘制测试准确率
    plt.title('Train History')  # 图表标题
    plt.ylabel('Accuracy')  # y轴标签
    plt.xlabel('Epoch')  # x轴标签
    plt.legend(['train', 'test'], loc='upper left')  # 图例
    plt.show()  # 显示图表


def main(data_dir):
    """
        主函数，执行数据集的加载、模型创建、训练和预测。

        参数：
        data_dir:数据集存储的目录路径。
    """
    if not all(os.path.exists(os.path.join(data_dir, f'data_batch_{i}')) for i in range(1, 6)) or \
       not os.path.exists(os.path.join(data_dir, 'test_batch')):
        print("Dataset not found.")
        return

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10(data_dir)

    x_val = x_train[:10000]
    y_val = y_train[:10000]
    x_train = x_train[10000:]
    y_train = y_train[10000:]

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    plot_cifar10_images(x_train, y_train, num_images=25)

    model = create_cnn_model()
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_history = train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=128)

    y_pred = model.predict(x_test)
    plot_images_labels_prediction(x_test, y_test, y_pred, start_index=0, num_images=10)

    show_train_history('accuracy', 'val_accuracy',train_history)
    show_train_history('loss', 'val_loss',train_history)


label_dict = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse",
    8: "ship", 9: "truck"
}

data_dir = '../images/CIFAR-10'  # 本地数据集路径

main(data_dir)




