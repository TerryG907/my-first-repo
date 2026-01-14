# CNN 图像二分类入门示例 (PyTorch)
# 目标：输入猫狗图片，输出预测概率
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# --------------------------
# 1️⃣ 数据预处理
# --------------------------
# 将图片缩放到 64x64，转 tensor，归一化
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 下载或读取数据集（这里用本地文件夹）
# 文件夹结构：
# data/train/cat/xxx.jpg
# data/train/dog/yyy.jpg
train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# --------------------------
# 2️⃣ 定义 CNN 模型
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 1)  # 输出层，二分类
        self.sigmoid = nn.Sigmoid()  # 输出概率

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 卷积 + ReLU + 池化
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


model = SimpleCNN()

# --------------------------
# 3️⃣ 定义损失函数和优化器
# --------------------------
criterion = nn.BCELoss()  # 二分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 4️⃣ 训练模型
# --------------------------
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)  # 转成 float 并调整维度 [batch,1]

        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算 loss
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# --------------------------
# 5️⃣ 测试模型准确率
# --------------------------
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs >= 0.5).float()  # 概率 >= 0.5 → 预测 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"测试集准确率: {accuracy * 100:.2f}%")




from PIL import Image
from torchvision import transforms

# 1️⃣ 加载图片
img = Image.open("data/test/dog/334685.jpg")

# 2️⃣ 和训练时同样预处理
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
img_tensor = transform(img).unsqueeze(0)  # 加 batch 维度 [1,C,H,W]

# 3️⃣ 模型预测
model.eval()
with torch.no_grad():
    output = model(img_tensor)
    predicted = 1 if output.item() >= 0.5 else 0

print("预测类别:", "狗" if predicted==1 else "猫")


Print("Who are you?")
