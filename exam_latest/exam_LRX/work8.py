# tensorboard --logdir=./logs 查看日志和图表
# 用resnet实现对CIFAR-10的分类

# 残差块的数量
# 代码中定义了 stack_n = 5，即每个阶段有 5 个残差块。这个 stack_n 控制每个阶段的深度。
# 每个阶段有 stack_n 个残差块，且有三个阶段：
# 第一个阶段：x = residual_block(x, 16, False)（重复 stack_n=5 次）
# 第二个阶段：x = residual_block(x, 32, True)（这个阶段中会有一次步长为 (2, 2) 的卷积操作，进行下采样）
# 第三个阶段：x = residual_block(x, 64, True)（同样进行下采样）
# 计算总层数
# 每个阶段的 stack_n = 5，每个残差块由两个卷积层和一个跳跃连接组成，所以每个阶段的卷积层数是 2 * stack_n。
# 但是，如果算上每个阶段的初始卷积层（在输入阶段进行的卷积）以及全连接层和池化层，总层数如下：
# 第一阶段的卷积层数：1 + 5（对应每个残差块的 2 层卷积，共 5 个残差块）= 1 + 5*2 = 11 层
# 第二阶段的卷积层数：1 + 5*2 = 11 层（同样的结构）
# 第三阶段的卷积层数：1 + 5*2 = 11 层（同样的结构）
# 最终的池化层 + 全连接层：1 层池化 + 1 层全连接
# 因此，网络的总层数大致为：
# 11 (第一阶段) + 11 (第二阶段) + 11 (第三阶段) + 2 (池化 + 全连接) = 35 层

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

# 1. 设置 TensorFlow 使用 GPU
def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        # 启用动态内存增长，避免一次性占满 GPU 内存
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is being used for computation.")
    else:
        print("No GPU found, running on CPU.")

# 2. CIFAR-10 数据加载和预处理
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


# 3. 定义残差块
def residual_block(x, o_filters, increase=False):
    stride = (1, 1)
    if increase:
        stride = (2, 2)

    o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
    conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(o1)
    o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
    conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(o2)
    if increase:
        projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(1e-4))(o1)
        block = add([conv_2, projection])
    else:
        block = add([conv_2, x])
    return block


# 4. 定义残差网络结构
def residual_network(img_input, classes_num=10, stack_n=5):
    # input: 32x32x3, output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(1e-4))(img_input)

    # 添加多个残差块
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16, output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32, output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    # BatchNorm + ReLU + GlobalAveragePooling
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-4))(x)
    return x


# 5. 定义训练过程中的学习率调度器
def scheduler(epoch):
    if epoch < 81:
        return 0.1
    elif epoch < 122:
        return 0.01
    return 0.001


# 6. 可视化训练过程（Loss 和 Accuracy 图表）
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


# 7. 可视化图像和标签对比
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


# 8. 设置数据增强
def setup_data_augmentation(x_train):
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)
    return datagen


# 9. 训练模型
def train_model(model, datagen, x_train, y_train, x_test, y_test, epochs, callbacks):
    history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=len(x_train) // 128,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    return history


# 10. 主函数
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

    # 进行预测并可视化
    predictions = model.predict(x_test[:25])
    plot_images_labels_prediction(x_test, y_test, predictions, start_index=0)


if __name__ == '__main__':
    main()
