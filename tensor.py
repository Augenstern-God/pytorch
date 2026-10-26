import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2

# 定义取出图片时要做的处理
image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# 创建训练集对象；首次运行需要联网下载
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=image_transform,
)
# image, label = training_data[0]

import matplotlib.pyplot as plt

# # 1. 取出第一个样本：图片和对应标签
# image, label = training_data[0]

# # 2. 查看图片的信息
# print("图片形状:", image.shape)
# print("类别标签:", label)

# # 3. 显示图片
# plt.imshow(image.squeeze(0), cmap="gray")
# plt.title(f"Label: {label}")
# plt.axis("off")
# plt.show()
from torch.utils.data import DataLoader

# 每批取 4 个样本，先按原顺序读取
train_loader = DataLoader(
    training_data,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

# 取出第一批
images, labels = next(iter(train_loader))

print("一批图片的形状:", images.shape)
print("一批标签的形状:", labels.shape)
print("这一批的标签:", labels)
# 类别编号 0～9 对应的名称
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.figure(figsize=(12, 3))

for i in range(len(images)):
    # 创建一行中的第 i+1 个显示位置
    plt.subplot(1, len(images), i + 1)

    # 显示这一批中的第 i 张图片
    plt.imshow(images[i].squeeze(0), cmap="gray")

    # 显示这张图片对应的类别
    label = labels[i].item()
    plt.title(f"{label}: {class_names[label]}")
    plt.axis("off")

plt.tight_layout()
plt.show()