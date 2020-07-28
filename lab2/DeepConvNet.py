import torch


def basic_block(in_c, out_c, k_size, activation=torch.nn.ReLU):
    layer0 = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c,
                             kernel_size=k_size, bias=True)
    layer1 = torch.nn.BatchNorm2d(num_features=out_c)
    layer2 = activation()
    layer3 = torch.nn.MaxPool2d(kernel_size=(1, 2))
    layer4 = torch.nn.Dropout(p=0.5)

    return torch.nn.Sequential(layer0, layer1, layer2, layer3, layer4)


class DeepConvNet(torch.nn.Module):
    def __init__(self, channels=2, cls_num=2, activation=torch.nn.ReLU):
        super(DeepConvNet, self).__init__()
        self.act_name = activation()._get_name()
        self.name = 'deepconvnet'
        self.first_conv = torch.nn.Conv2d(in_channels=1, out_channels=25,
                                          kernel_size=(1, 5), bias=True)
        self.block0 = basic_block(in_c=25, out_c=25,
                                  k_size=(channels, 1), activation=activation)
        self.block1 = basic_block(in_c=25, out_c=50,
                                  k_size=(1, 5), activation=activation)
        self.block2 = basic_block(in_c=50, out_c=100,
                                  k_size=(1, 5), activation=activation)
        self.block3 = basic_block(in_c=100, out_c=200,
                                  k_size=(1, 5), activation=activation)
        self.dense = torch.nn.Linear(in_features=8600, out_features=cls_num)

    def forward(self, x):
        # x = torch.unsqueeze(x, 1)
        x = self.first_conv(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        # x = torch.softmax(x, dim=1)
        return x
