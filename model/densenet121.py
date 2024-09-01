import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal

__all__ = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet264']

'''-------------一、构造初始卷积层-----------------------------'''
def Conv1(in_planes, places, stride=2):
    return nn.SequentialCell(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, pad_mode='pad', padding=3, has_bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU()
    )

'''-------------二、定义Dense Block模块-----------------------------'''

'''---（1）构造Dense Block内部结构---'''
class _DenseLayer(nn.Cell):
    def __init__(self, in_planes, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.SequentialCell([
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        ])
        self.dropout = nn.Dropout(keep_prob=1.0 - drop_rate) if drop_rate > 0 else None

    def construct(self, x):
        y = self.dense_layer(x)
        if self.dropout:
            y = self.dropout(y)
        return ops.concat((x, y), 1)

'''---（2）构造Dense Block模块---'''
class DenseBlock(nn.Cell):
    def __init__(self, num_layers, in_planes, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_planes + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)

'''-------------三、构造Transition层-----------------------------'''
class _TransitionLayer(nn.Cell):
    def __init__(self, in_planes, out_planes):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.SequentialCell([
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, has_bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ])

    def construct(self, x):
        return self.transition_layer(x)

'''-------------四、搭建DenseNet网络-----------------------------'''
class DenseNet(nn.Cell):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=10):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=1, places=init_channels)

        num_features = init_channels

        self.layer1 = DenseBlock(num_layers=blocks[0], in_planes=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(in_planes=num_features, out_planes=num_features // 2)
        num_features = num_features // 2

        self.layer2 = DenseBlock(num_layers=blocks[1], in_planes=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(in_planes=num_features, out_planes=num_features // 2)
        num_features = num_features // 2

        self.layer3 = DenseBlock(num_layers=blocks[2], in_planes=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(in_planes=num_features, out_planes=num_features // 2)
        num_features = num_features // 2

        self.layer4 = DenseBlock(num_layers=blocks[3], in_planes=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Dense(num_features, num_classes, weight_init=HeNormal())

    def construct(self, x):
        x = x.astype(ms.float32)  # 将输入转换为 float32 类型
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        x = ops.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

def DenseNet121():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16])

def DenseNet169():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 32, 32])

def DenseNet201():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 48, 32])

def DenseNet264():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 64, 48])
