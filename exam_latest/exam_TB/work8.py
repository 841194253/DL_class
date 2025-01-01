
# 1. 模型概述
# 代码中使用了 VGG19 网络结构作为预训练模型，通过 tensorflow.keras.applications.VGG19 导入。VGG19 是一个深度卷积神经网络（CNN），具有 19 层，并且通常用于图像分类任务，能够有效提取图像的特征。在这个代码中，VGG19 作为预训练模型（使用 ImageNet 数据集的权重），然后添加了一个新的全连接层，适应 CIFAR-10 数据集。
# 2. 数据加载和预处理
# 使用了 CIFAR-10 数据集。数据集通过 pickle 文件加载，并进行了预处理：
# 数据被转换为适合 VGG19 输入的格式（32x32 RGB 图像）。
# 数据进行了归一化处理，将像素值从 [0, 255] 范围压缩到 [0, 1] 范围。
# 标签通过 to_categorical 进行了 One-Hot 编码，确保每个标签是一个独热向量（用于多分类问题）。
# 3. 数据增强
# 为了提高模型的泛化能力，使用了 ImageDataGenerator 进行数据增强：
# 随机水平翻转。
# 随机平移（宽度和高度方向各有 12.5% 的偏移）。
# 4. VGG19 模型构建
# 在 build_vgg19_model() 函数中，使用了 VGG19 作为基础模型，并且：
# 不包含原始模型的顶层（include_top=False），这样可以重新定义输出层。
# 基础 VGG19 模型的卷积层被冻结（layer.trainable = False），这意味着这些层的权重在训练过程中不会被更新，以利用 ImageNet 训练时学到的特征。
# 新的全连接层（Dense）被添加到模型的末尾，用于对 CIFAR-10 的 10 类进行分类。
# 5. 模型训练
# 使用 SGD（随机梯度下降）优化器进行训练。
# 采用了 EarlyStopping 回调，监控验证损失，避免过拟合，并在模型不再改进时停止训练。
# ModelCheckpoint 回调用于保存最佳的模型权重（基于验证准确率）。
# 6. 模型评估与可视化
# 模型训练完成后，通过 model.evaluate() 函数在测试集上评估模型性能，并输出测试集的准确率。
# 绘制了混淆矩阵和 ROC 曲线：
# 混淆矩阵：显示预测与真实标签之间的关系，用于评估分类器的表现。
# ROC 曲线：展示了不同分类阈值下的假阳性率与真正阳性率，评估多类分类任务的性能。
# 7. 可视化训练过程
# 使用 matplotlib 绘制了训练和验证过程中损失（Loss）和准确率（Accuracy）的变化曲线，帮助分析模型的学习进程。
# 8. 可视化预测结果
# 从测试集中随机选择 10 张图片，展示它们的真实标签与预测标签，并可视化图像。
# 9. 主要函数解释
# load_and_preprocess_cifar10：加载并预处理 CIFAR-10 数据集。
# create_data_generator：初始化数据增强。
# build_vgg19_model：构建 VGG19 模型，并添加新层来适应 CIFAR-10 分类任务。
# train_model：训练模型并保存最佳模型权重。
# evaluate_model：评估模型在测试集上的性能。
# plot_confusion_matrix 和 plot_roc_curve：可视化混淆矩阵和 ROC 曲线。
# plot_training_history：可视化训练过程中损失和准确率的变化。
# visualize_predictions：显示预测结果，比较预测标签和真实标签。
# 10. 改进建议
# 数据增强改进：当前的增强方式相对简单，可以尝试更多种类的数据增强技术，例如旋转、缩放、颜色调整等，以提高模型的泛化能力。
# VGG19 微调：虽然当前冻结了 VGG19 的卷积层，微调部分卷积层可能会进一步提高模型在 CIFAR-10 上的性能。微调通常通过解冻最后几层进行。
# 优化器调整：可以尝试使用其他优化器（如 Adam 或 RMSprop），以提高模型的收敛速度和准确度。
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

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

# 构建VGG19模型并定义新顶层
def build_vgg19_model(input_shape=(32, 32, 3), num_classes=10):
    from tensorflow.keras.applications import VGG19

    # 使用VGG19作为基础模型，不包括顶层（include_top=False），并加载ImageNet的预训练权重
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # 冻结预训练层
    for layer in base_model.layers:
        layer.trainable = False

    # 构建新模型
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # 编译模型
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 训练模型
def train_model(model, datagen, x_train, y_train, x_test, y_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=50,
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

    test_losses = history.history['val_loss']
    test_accuracies = history.history['val_accuracy']

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

    # 构建VGG19模型
    model = build_vgg19_model(input_shape=(32, 32, 3), num_classes=10)

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
