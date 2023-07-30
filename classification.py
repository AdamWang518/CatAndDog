import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from model import discriminator
transform=transforms.Compose([
   transforms.Resize((256,256)),
    transforms.ToTensor()
]
)
trainset=ImageFolder('D:\\CatAndDog\\reduce',transform=transform)
trainloader=DataLoader(trainset,batch_size=32, shuffle=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# print(trainset.classes)
model = discriminator().to(device)
# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 訓練模型
total_epochs = 50
for epoch in range(total_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播和參數更新
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(i)
    print(
        f"Epoch [{epoch+1}/{total_epochs}], Loss: {running_loss/len(trainloader)}")
    if (epoch + 1) % 5 == 0:
        torch.save(model, f'Discriminator_epoch_{epoch}.pth')
        print('Model saved.')
