import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(800, 100)
        #  # 800 = ((((28 - 5 + 1) / 2) - 5 + 1) / 2) * ((((28 - 5 + 1) / 2) - 5 + 1) / 2) * 50
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    layer = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1).to(device)

    print(layer) # Conv2d(1, 20, kernel_size=(3, 3), stride=(1, 1))

    # 초기 weight 값
    print(type(layer.weight)) # <class 'torch.nn.parameter.Parameter'>

    # weight를 numpy로 만들려면 detach가 필요
    print(layer.weight.detach().cpu().numpy().shape) # (20, 1, 3, 3)

    # MNISt trina data 가져오기
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size = 32
    )

    # model 정의
    model = CnnNet().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # train
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # loss 총합 계산
            total_loss += loss.item()

        print('epoch : {} , loss : {:.4f}'.format(epoch+1, total_loss))
    
    # MNIST test data 가져오기
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size = 32
    )

    # 예측
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item()

            pred = output.argmax(dim = 1, keepdim = True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    print('loss : {:.4f}, accuracy : {:.2f}'.format(test_loss / len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))


