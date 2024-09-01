import mindspore.nn as nn
import mindspore as ms

class VGG16(nn.Cell):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(1, 64, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.SequentialCell(
            nn.Dense(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(256, num_classes)
        )

    def construct(self, x):
        x = x.astype(ms.float32)  # 将输入转换为 float32 类型
        x = self.features(x)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.classifier(x)
        return x