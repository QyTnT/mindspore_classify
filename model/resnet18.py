import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform
from mindspore import Tensor
import mindspore.numpy as np
import mindspore as ms


class Resnet18(nn.Cell):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()

        self.model0 = nn.SequentialCell([
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=2, pad_mode='pad', padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, pad_mode='pad', padding=1)
        ])

        self.model1 = nn.SequentialCell([
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])

        self.R1 = nn.ReLU()

        self.model2 = nn.SequentialCell([
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])

        self.R2 = nn.ReLU()

        self.model3 = nn.SequentialCell([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])

        self.en1 = nn.SequentialCell([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, pad_mode='pad', padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])

        self.R3 = nn.ReLU()

        self.model4 = nn.SequentialCell([
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])

        self.R4 = nn.ReLU()

        self.model5 = nn.SequentialCell([
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])

        self.en2 = nn.SequentialCell([
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, pad_mode='pad', padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])

        self.R5 = nn.ReLU()

        self.model6 = nn.SequentialCell([
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])

        self.R6 = nn.ReLU()

        self.model7 = nn.SequentialCell([
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])

        self.en3 = nn.SequentialCell([
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, pad_mode='pad', padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])

        self.R7 = nn.ReLU()

        self.model8 = nn.SequentialCell([
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])

        self.R8 = nn.ReLU()

        self.aap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels=512, out_channels=num_classes)

    def construct(self, x):
        x = x.astype(ms.float32)  # 将输入转换为 float32 类型
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
