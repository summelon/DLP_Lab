import torch


def first_conv():
    layer0 = torch.nn.Conv2d(
            in_channels=1, out_channels=16,
            kernel_size=(1, 51), padding=(0, 25), bias=False)
    layer1 = torch.nn.BatchNorm2d(num_features=16)

    return torch.nn.Sequential(layer0, layer1)


def depthwise_conv(activation=torch.nn.ReLU):
    layer0 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                             kernel_size=(2, 1), groups=16, bias=False)
    layer1 = torch.nn.BatchNorm2d(num_features=32)
    layer2 = activation()
    layer3 = torch.nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
    layer4 = torch.nn.Dropout(p=0.25)

    return torch.nn.Sequential(layer0, layer1, layer2, layer3, layer4)


def separable_conv(activation=torch.nn.ReLU):
    layer0 = torch.nn.Conv2d(in_channels=32, out_channels=32,
                             kernel_size=(1, 15), padding=(0, 7), bias=False)
    layer1 = torch.nn.BatchNorm2d(num_features=32)
    layer2 = activation()
    layer3 = torch.nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
    layer4 = torch.nn.Dropout(p=0.25)

    return torch.nn.Sequential(layer0, layer1, layer2, layer3, layer4)


class EEGNet(torch.nn.Module):
    def __init__(self, activation=torch.nn.ReLU):
        super(EEGNet, self).__init__()
        self.act_name = activation()._get_name()
        self.name = 'eegnet'
        self.firstConv = first_conv()
        self.depthwiseConv = depthwise_conv(activation=torch.nn.ReLU)
        self.separableConv = separable_conv(activation=torch.nn.ReLU)
        self.classify = torch.nn.Sequential(
                torch.nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classify(x)
        return x


def main():
    import dataloader
    model = EEGNet()
    model.eval()
    print(model)
    data, _, _, _ = dataloader.read_bci_data('./dataset')
    x = torch.Tensor(data[0:1])
    print(model(x))


if __name__ == "__main__":
    main()
