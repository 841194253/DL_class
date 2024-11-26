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
