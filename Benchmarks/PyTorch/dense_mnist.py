import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
sys.path.append("../..")

from Models.PyTorch import dense

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") CUDA cc 3.5 not suported
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
            transforms.Normalize((0,), (255.,))
    ])

mnist_dataset = datasets.MNIST(".", train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=32, shuffle=True)

model = dense.DenseModel(inputs=784, outputs=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 200
for epoch in range(1, num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))