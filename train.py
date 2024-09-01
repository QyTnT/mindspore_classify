import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
from mindspore import context, Tensor, Model
from mindspore.nn import Accuracy, Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from mindspore.train.callback import Callback
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import seaborn as sns


def get_result(y_test=None, y_pred=None, name=None):
    # 计算准确率、精确率、召回率和 F1 值
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print()
    print(f"{name}_accuracy:", round(accuracy, 2), '\n')
    #         print(f"{name}_precision:", round(precision,2),'\n')
    #         print(f"{name}_recall:", round(recall,2),'\n')
    #         print(f"{name}_f1:", round(f1,2),'\n')

    con_mat = confusion_matrix(y_test, y_pred)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)  # np.around(): 四舍五入

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), facecolor='w')

    # 绘制混淆矩阵
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='Greens', ax=axs[0])
    axs[0].set_title(f'{name}_Confusion_Matrix')
    axs[0].set_xlabel('Predict')
    axs[0].set_ylabel('True')

    # 绘制归一化后的混淆矩阵
    #     plt.delaxes(axs[1])  # 删除 第二个 子图 不会产生 画布 重要

    sns.heatmap(con_mat_norm, annot=True, fmt='.2f', cmap='Greens', ax=axs[1])
    axs[1].set_title(f'{name}_Normalized_Confusion_Matrix')
    axs[1].set_xlabel('Predict')
    axs[1].set_ylabel('True')

    # 保存图像
    #     fig.savefig(f'images/{name}_Confusion_Matrix.png')
    plt.show()
    return accuracy, precision, recall, f1

def get_acc_loss(train_epoch_acc=None,val_epoch_acc=None,train_epoch_loss=None,val_epoch_loss=None):
    # 绘制损失和准确率曲线
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_epoch_acc, 'o-', c='b', label='train')
    plt.plot(val_epoch_acc, '*-', c='r', label='val')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.title('train_val_acc')
    plt.subplot(1, 2, 2)
    plt.plot(train_epoch_loss, 'o-', c='b', label='train')
    plt.plot(val_epoch_loss, '*-', c='r', label='val')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('train_val_loss')
    # plt.yticks(range(0,1,0.2))
    plt.tight_layout()
    # 保存图片
    # plt.savefig(f'images/{k}_loss_accuracy_plot.png')
    # plt.savefig('images/'+k+'_loss_accuracy_plot.png')
    plt.show()
    return None
class EpochLossMonitor(Callback):
    def __init__(self, model, loss_fn, train_dataset, val_dataset):
        super(EpochLossMonitor, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epoch_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        # 计算训练集损失和精度
        train_loss, train_acc = self.calculate_metrics(self.train_dataset)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        # 计算验证集损失和精度
        val_loss, val_acc = self.calculate_metrics(self.val_dataset)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        print(f"Epoch {cb_params.cur_epoch_num}, "
              f"Train Loss: {train_loss}, Train Accuracy: {train_acc}, "
              f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    def calculate_metrics(self, dataset):
        """计算给定数据集的损失和精度"""
        total_loss = 0
        correct = 0
        total = 0
        for data in dataset.create_dict_iterator():
            inputs = data['data']
            labels = data['label']
            outputs = self.model.predict(inputs)
            loss = self.loss_fn(outputs, labels)
            total_loss += loss.asnumpy().sum()
            predicted = outputs.asnumpy().argmax(axis=1)
            correct += (predicted == labels.asnumpy()).sum()
            total += labels.size

        accuracy = correct / total
        average_loss = total_loss / total
        return average_loss, accuracy

    def get_epoch_losses(self):
        return self.epoch_losses

    def get_train_accuracies(self):
        return self.train_accuracies

    def get_val_accuracies(self):
        return self.val_accuracies

    def get_train_losses(self):
        return self.train_losses

    def get_val_losses(self):
        return self.val_losses


# 设置运行模式和设备
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # 可以将 "CPU" 替换为 "GPU" 或 "Ascend"

def create_dataset(data, labels, batch_size=32, shuffle=True):
    dataset = ds.NumpySlicesDataset({"data": data, "label": labels}, shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
# 数据文件路径
data_path = "./semeion+handwritten+digit/semeion.data"
# 读取数据
def load_semeion_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 初始化数据列表
    X = []
    y = []
    # 处理每一行数据
    for line in lines:
        parts = line.split()  # 以空格分割每一行
        features = list(map(float, parts[:256]))  # 将前256个特征转换为浮点数
        label = list(map(float, parts[256:]))  # 将最后一个元素转换为整数作为标签
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)
X,y = load_semeion_data(data_path)
X,y = X.reshape(-1,1,16,16),np.argmax(y,axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,shuffle=True)
train_dataset = create_dataset(X_train, Y_train)
val_dataset = create_dataset(X_test, Y_test)

from model.vgg16 import VGG16
from model.resnet18 import Resnet18
from model.densenet121 import DenseNet121
# 初始化模型
# net = VGG16(num_classes=10)
# net = Resnet18(num_classes=10)
model_name = 'DenseNet121'
net = DenseNet121()

# 加载预训练模型
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)
model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={"Accuracy": Accuracy()})

config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
ckpoint = ModelCheckpoint(prefix=model_name, directory="./weights", config=config_ck)
lossmonitor = EpochLossMonitor(model,criterion,train_dataset,val_dataset,)
epoch = 5
model.train(epoch, train_dataset, callbacks=[ckpoint, lossmonitor])
train_epoch_acc = lossmonitor.get_train_accuracies()
val_epoch_acc = lossmonitor.get_val_accuracies()
train_epoch_loss = lossmonitor.get_train_losses()
val_epoch_loss = lossmonitor.get_val_losses()
get_acc_loss(train_epoch_acc,val_epoch_acc,train_epoch_loss,val_epoch_loss)


true = []
pred = []
# 在推理过程中关闭梯度计算
for data in val_dataset.create_dict_iterator():
    inputs = data['data']
    labels = data['label']

    # 转换为 MindSpore Tensor 并将其移至计算设备
    inputs = Tensor(inputs)
    labels = Tensor(labels)

    # 获取模型的输出
    outputs = model.predict(inputs)

    # 获取最大概率的索引
    predicted = outputs.asnumpy().argmax(axis=1)

    # 将结果存储到列表中
    true.append(labels.asnumpy())
    pred.append(predicted)

# 将 true 和 pred 列表中的结果拼接
y_test = np.concatenate(true, axis=0)
y_pred = np.concatenate(pred, axis=0)
# 现在 y_test 和 y_pred 包含了验证集的真实标签和模型预测的标签

vgg16_acc= get_result(y_test=y_test,y_pred=y_pred,name = 'vgg16')
