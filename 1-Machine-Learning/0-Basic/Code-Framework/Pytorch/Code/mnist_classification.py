import torchvision
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64
n_epochs = 5

################ load data ################
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5,std=0.5)])

data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=batch_size,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=batch_size,
                                               shuffle=True)

################ define model ################


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(
                                             64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x


################ training set ################
model = Model()
model.to(device)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

################ training ################
for epoch in range(n_epochs):

    start = time.time()

    # 1.训练模式让 dropout生效
    model.train()
    loss_train_sum = 0.0

    print(f'Epoch {epoch + 1}/{n_epochs}'.center(40,'-'))

    for step_train, (features, labels) in enumerate(data_loader_train, 1):

        features = features.to(device)
        labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()
        outputs = model(features)
        loss_train = loss_func(outputs, labels)

        # 反向传播求梯度
        loss_train.backward()
        optimizer.step()

        loss_train_sum += loss_train.item()
        if step_train % 100 == 0:
            print(
                f'step = {step_train}, loss = {loss_train_sum / step_train:.3f}')

    # 2.测试模式让 dropout 不生效
    model.eval()
    loss_test_sum = 0.0

    for step_test, (features, labels) in enumerate(data_loader_test, 1):

        features = features.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(features)
            loss_test = loss_func(outputs, labels)

        loss_test_sum += loss_test.item()

    print(f'Time: {time.time() - start}, Train_Loss: {loss_train_sum/step_train:.4f}, Test_Loss:{loss_test_sum/step_test:.4f}')

################ save model ################
torch.save(model, "model.pkl")
