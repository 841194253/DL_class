# 数据加载与预处理
# 代码中通过 load_and_preprocess_cifar10() 函数加载 CIFAR-10 数据集，并进行以下处理：
# 加载训练集（5个批次）和测试集数据。
# 转换数据形状为 32x32x3。
# 对数据进行归一化处理，将像素值缩放到 [0, 1] 范围。
# 使用 to_categorical() 将标签进行 One-Hot 编码。

# 迁移学习模型构建与微调
# 代码使用了 VGG16 预训练模型，并冻结了预训练层以提取特征，然后构建了新顶层来适配 CIFAR-10 数据集的分类任务。

# 模型训练
# 代码使用 ImageDataGenerator 进行数据增强，并通过 EarlyStopping 和 ModelCheckpoint 来防止过拟合并保存最优模型。

# 分类准确率
# 训练完成后，通过 evaluate_model() 函数评估模型在测试集上的性能，输出准确率。

# 混淆矩阵与ROC曲线
# 混淆矩阵和ROC曲线的可视化分别使用 plot_confusion_matrix() 和 plot_roc_curve() 函数

# 可视化预测结果
# 通过 visualize_predictions() 可视化一些预测结果，显示真实标签与模型预测标签，帮助分析错误分类样本。

# 迁移学习模型的表现及影响因素
# 数据规模：CIFAR-10 数据集较小，可能导致过拟合。通过数据增强和早停策略可以缓解过拟合。
# 预训练模型选择：使用了 VGG16 作为预训练模型，适用于图像分类任务，但对于较小图像（如32x32），可能没有完全发挥其潜力。更适合使用轻量级模型如 MobileNet 或 ResNet。
# 改进方向
# 调整超参数：优化学习率、批大小、训练轮数等超参数，可能提高模型表现。
# 数据增强策略：可以增加更多的增强方式，如旋转、缩放等。
# 细调预训练模型：允许更多的预训练层微调，而不是只冻结预训练层，可能帮助模型更好地适应特定任务。

import os
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# CIFAR-10 标签映射字典
label_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# 加载并预处理 CIFAR-10 数据集（本地读取）
def load_and_preprocess_cifar10(data_dir):
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

    # 数据拼接
    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)
    x_test = np.concatenate(x_test).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)

    # 数据归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-Hot Encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

# 数据增强
def create_data_generator(x_train):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.125,
        height_shift_range=0.125
    )
    datagen.fit(x_train)
    return datagen

# 构建VGG16模型并定义新顶层
def build_vgg16_model(input_shape=(32, 32, 3), num_classes=10):
    from tensorflow.keras.applications import VGG16

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 训练模型
def train_model(model, datagen, x_train, y_train, x_test, y_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=3,
        validation_data=(x_test, y_test),
        callbacks=[early_stop, checkpoint]
    )

    return history

# 评估模型
def evaluate_model(model, x_test, y_test):
    model.load_weights('best_model.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    return test_loss, test_acc

# 绘制混淆矩阵
def plot_confusion_matrix(cm, num_classes=10, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(num_classes), [label_dict[i] for i in range(num_classes)])
    plt.yticks(np.arange(num_classes), [label_dict[i] for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    thresh = cm.max() / 2.0  # 色彩阈值控制显示的文本颜色
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 绘制多类 ROC 曲线
def plot_roc_curve(y_true, y_pred, num_classes=10, save_path='roc_curve.png'):
    y_true = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label_dict[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 可视化训练过程（Loss 和 Accuracy 图表）
def plot_training_history(history, model, x_test, y_test):
    # 创建一个大图，包含训练和测试图表
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 训练 Loss 和 Accuracy 图表（左图）
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Training Loss and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Training Accuracy and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.savefig('train.png')
    plt.show()

    # 获取每个 epoch 后的测试集损失和准确率
    test_losses = []
    test_accuracies = []

    for epoch in range(len(history.history['loss'])):
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # 测试 Loss 和 Accuracy 图表（右图）
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].plot(test_losses, label='Test Loss', color='r')
    axes[0].set_title('Test Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(test_accuracies, label='Test Accuracy', color='g')
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.savefig('test.png')
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


# 可视化预测结果
def visualize_predictions(model, x_test, y_test, label_dict, num_samples=10):
    # 随机选择一些测试样本
    idx = np.random.choice(x_test.shape[0], num_samples, replace=False)

    # 创建一个图像
    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(idx):
        # 获取样本图像和真实标签
        img = x_test[idx]
        true_label = np.argmax(y_test[idx])

        # 获取模型预测标签
        pred_label = np.argmax(model.predict(np.expand_dims(img, axis=0)))

        # 显示图像
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')

        # 设置标题：显示真实标签和预测标签
        title = f"True: {label_dict[true_label]}\nPred: {label_dict[pred_label]}"
        plt.title(title, fontsize=12)

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    data_dir = '../../images/CIFAR-10/'  # 你本地数据集存放的目录
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10(data_dir)

    # 数据增强
    datagen = create_data_generator(x_train)

    # 构建VGG16模型
    model = build_vgg16_model(input_shape=(32, 32, 3), num_classes=10)

    # 训练模型
    history = train_model(model, datagen, x_train, y_train, x_test, y_test)

    # 评估模型
    test_loss, test_acc = evaluate_model(model, x_test, y_test)

    # 绘制训练过程（Loss 和 Accuracy 图表）
    plot_training_history(history, model, x_test, y_test)

    # 绘制评估结果（混淆矩阵和ROC曲线）
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_confusion_matrix(cm)

    # 绘制ROC曲线
    plot_roc_curve(y_true_classes, y_pred)

    # 可视化预测结果，显示10个样本
    visualize_predictions(model, x_test, y_test, label_dict, num_samples=10)


if __name__ == "__main__":
    main()
