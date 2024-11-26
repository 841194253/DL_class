# tensorboard --logdir=./logs 查看日志和图表
# 用resnet实现对CIFAR-10的分类

# 网络结构
# 在 ResNet-32 中，每个阶段（Stage）由多个残差块（Residual Block）组成。ResNet-32 的典型配置如下：
# 第一阶段：输入卷积层
# 第二阶段：堆叠多个残差块
# 第三阶段：下采样并增加通道数
# 最终层：池化和全连接层
# 每一层的作用
# 卷积层：每个卷积层都会增加网络的深度。
# 残差块：每个残差块通常由两层卷积层组成，且每个残差块的输入和输出维度相同。
# 每个阶段的堆叠数：在 ResNet-32 中，每个阶段包含一定数量的残差块，通常每个阶段的残差块数量相同。
# ResNet-32 具体层数计算
# 网络结构如下：
# 第一层（输入卷积层）：1 层。
# 每个阶段：每个残差块由 2 层卷积构成，因此一个残差块有 2 层卷积，多个残差块的总层数为 2 * 残差块数。
# 在 ResNet-32 中，第一阶段有 5 个残差块，因此有 5 * 2 = 10 层卷积。
# 第二阶段有 5 个残差块，因此也是 5 * 2 = 10 层卷积。
# 第三阶段有 5 个残差块，也是 5 * 2 = 10 层卷积。
# 全连接层：1 层。
# 计算总层数
# 输入卷积层：1 层
# 每个阶段的残差块：每个残差块有 2 层卷积，堆叠 5 个残差块时，共有 5 * 2 = 10 层卷积
# 最终的全连接层：1 层
# 因此，ResNet-32 的总层数为：
# 1（输入卷积层）
# 10（第一阶段的 5 个残差块，每个残差块有 2 层卷积）
# 10（第二阶段的 5 个残差块，每个残差块有 2 层卷积）
# 10（第三阶段的 5 个残差块，每个残差块有 2 层卷积）
# 1（全连接层）
# 总层数 = 1 + 10 + 10 + 10 + 1 = 32 层
# ResNet-32 的层数为 32 层，其中：
# 1 层输入卷积层
# 3 个阶段，每个阶段 5 个残差块，每个残差块由 2 层卷积构成，总共有 30 层卷积层
# 1 层全连接层

# 迁移学习是一种机器学习方法，它将一个任务上训练得到的知识应用到另一个相关任务上。通常情况下，迁移学习通过使用在大型数据集（如ImageNet）上预训练的模型，并将其应用于新的、数据较少的任务上，显著提高新任务的学习效果。迁移学习的基本步骤包括：
# 选择预训练模型：在一个大规模数据集上训练得到的模型（如ResNet、VGG、Inception等），这些模型已经学会了有用的特征表示。
# 冻结部分层：将预训练模型的底层（通常是特征提取层）冻结，这些层捕捉到了低层次的特征（如边缘、纹理等），不需要重新训练。
# 微调：只训练模型的顶层（通常是分类层或全连接层），使其适应新任务的特定类别。
# 为什么选择迁移学习：
# 数据不足：迁移学习非常适合数据量较小的场景，因为预训练模型已经在大量数据上学习了有效的特征，可以有效地帮助新任务，即使新任务的数据较少。
# 小样本问题：当目标任务的数据样本较少时，直接训练一个深度学习模型往往会导致过拟合。而迁移学习能够有效避免这一问题，因为预训练模型已经在大规模数据上学到了一些通用的特征。

# 利用预训练模型特征提取：使用 residual_network 构建 ResNet 模型，适用于 CIFAR-10 数据集。
# 全连接层输出被设为 10 类（CIFAR-10 分类任务）。

# 微调模型：
# 使用了学习率调度（scheduler）和 SGD 优化器结合 Nesterov 动量，能有效加速收敛。
# 数据增强器 (setup_data_augmentation) 提供了水平翻转和位移变换。
# 使用了 EarlyStopping 回调函数以防止过拟合。

# 结果分析
# 模型表现影响因素：
# 数据规模：CIFAR-10 数据集较小，易过拟合。
# 预训练模型选择：ResNet 架构更适合图像分类任务。
#
# 改进方向：
# 增强数据：更多数据增强策略。
# 调整超参数：进一步优化学习率、batch size。
# 引入更深层的网络（如 ResNet-50 或 ResNet-101）。
# 使用更多现代优化器（如 AdamW）试试较低学习率。
# 数据增强策略可以增加更多变化，如随机旋转、对比度调整等。

import os
import numpy as np
import pickle
import keras
from keras import optimizers
from keras.layers import Conv2D, Dense, Input, BatchNormalization, Activation, add, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, Callback
from keras.utils import to_categorical
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import time

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred):
    """
    绘制ROC曲线。
    """
    plt.figure(figsize=(10, 8))

    # 针对每一类分别绘制ROC曲线
    for i in range(len(label_dict)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label_dict[i]} (AUC = {roc_auc:.2f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    """
    绘制混淆矩阵。
    """
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_dict.values()))

    plt.figure(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.show()


# from keras.applications import ResNet50
# from keras.layers import Flatten, Dense, GlobalAveragePooling2D
# from keras.models import Model


# # 加载预训练的 ResNet50 模型（不包含顶部的全连接层）
# def create_pretrained_model(input_shape=(32, 32, 3), num_classes=10):
#     # 加载 ResNet50 预训练模型，输入尺寸与 CIFAR-10 图像匹配
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#
#     # 冻结底层卷积层，不训练这些层
#     base_model.trainable = False
#
#     # 添加新的顶部结构，用于CIFAR-10分类
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)  # 使用全局平均池化
#     x = Dense(1024, activation='relu')(x)  # 添加全连接层
#     x = Dense(num_classes, activation='softmax')(x)  # 输出层，适配CIFAR-10
#
#     # 构建最终模型
#     model = Model(inputs=base_model.input, outputs=x)
#
#     # 编译模型
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     return model

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

# 设置 TensorFlow 使用 GPU
def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        # 启用动态内存增长，避免一次性占满 GPU 内存
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is being used for computation.")
    else:
        print("No GPU found, running on CPU.")

# CIFAR-10 数据加载和预处理
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


# 定义残差块
def residual_block(x, filters, stride=1, increase=False):
    """
    定义一个残差块。

    参数：
    x: 输入张量
    filters: 输出卷积层的通道数
    stride: 步幅
    increase: 是否进行下采样

    返回：
    处理后的输出张量
    """
    shortcut = x

    # 第一层卷积
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same',
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    # 第二层卷积
    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization(momentum=0.9)(x)

    # 如果需要增加通道数，则进行下采样
    if increase:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same',
                          kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(shortcut)

    # 合并 shortcut 和卷积后的输出
    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


# 定义残差网络结构
def residual_network(img_input, classes_num=10, stack_n=5):
    """
    构建 ResNet-32 网络结构。

    参数：
    img_input: 输入层
    classes_num: 分类类别数
    stack_n: 每个阶段堆叠的残差块数（5）

    返回：
    网络输出
    """
    # 第一层卷积
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(img_input)

    # 第一个阶段，堆叠 5 个残差块
    for _ in range(stack_n):
        x = residual_block(x, 16)

    # 第二个阶段，增加通道数并下采样
    x = residual_block(x, 32, stride=2, increase=True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32)

    # 第三个阶段，增加通道数并下采样
    x = residual_block(x, 64, stride=2, increase=True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64)

    # BatchNorm + ReLU + GlobalAveragePooling
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-4))(x)

    return x


# 定义训练过程中的学习率调度器
def scheduler(epoch):
    if epoch < 81:
        return 0.1
    elif epoch < 122:
        return 0.01
    return 0.001


# 可视化训练过程（Loss 和 Accuracy 图表）
def plot_training_history(history):
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

    # 保存训练 Loss 和 Accuracy 图表（保存为 train.png）
    plt.savefig('train.png')
    plt.show()

    # 创建一个新的图像，只包含测试图表
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 测试 Loss 和 Accuracy 图表（左图）
    axes[0].plot(history.history['val_loss'], label='Test Loss')
    axes[0].set_title('Test Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['val_accuracy'], label='Test Accuracy')
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # 保存测试 Loss 和 Accuracy 图表（保存为 test.png）
    plt.savefig('test.png')
    plt.show()


# 可视化图像和标签对比
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


# 设置数据增强
def setup_data_augmentation(x_train):
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)
    return datagen

# 训练模型
def train_model(model, datagen, x_train, y_train, x_test, y_test, epochs, callbacks):
    history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=len(x_train) // 128,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    return history


# 主函数
def main():
    # 设置 GPU
    setup_gpu()

    # 加载和预处理 CIFAR-10 数据
    data_dir = '../../images/CIFAR-10/'  # CIFAR-10 数据集路径
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10(data_dir)

    # 定义输入和模型
    img_input = Input(shape=(32, 32, 3))
    output = residual_network(img_input)
    model = Model(img_input, output)

    # 编译模型
    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 设置回调函数
    tb_cb = TensorBoard(log_dir='./logs', histogram_freq=0)
    change_learning_rate = LearningRateScheduler(scheduler)
    cbks = [change_learning_rate, tb_cb]

    # 数据增强
    datagen = setup_data_augmentation(x_train)

    # 定义早停机制
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 可以选择 'val_loss' 或 'val_accuracy'
        patience=10,  # 如果验证集性能连续 10 个 epoch 无提升，则停止训练
        verbose=1,  # 输出日志
        restore_best_weights=True  # 恢复到最好的权重
    )

    callbacks = [early_stopping, change_learning_rate, tb_cb, cbks]

    # 记录训练的开始时间
    start_time = time.time()

    # 训练模型
    epochs = 50
    history = train_model(model, datagen, x_train, y_train, x_test, y_test, epochs, callbacks)

    # 记录训练的结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time  # 训练所用时间（秒）
    print(f"Training time: {elapsed_time / 60:.2f} minutes")

    # 保存训练过程图表
    plot_training_history(history)

    # 保存模型
    model.save('resnet_model.h5')

    # subset_size = 25  # 选择部分数据

    # 使用完整数据进行预测
    # 进行预测并可视化
    predictions = model.predict(x_test)
    plot_images_labels_prediction(x_test, y_test, predictions, start_index=0)

    # 混淆矩阵
    plot_confusion_matrix(y_test, predictions)

    # ROC曲线
    plot_roc_curve(y_test, predictions)


if __name__ == '__main__':
    main()
