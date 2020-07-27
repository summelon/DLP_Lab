import torch


def basic_block(in_c, out_c):
    layer0 = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c,
                             kernel_size=(), bias=True)


class DeepConvNet(torch.nn.Module):
    def __init(self):
        super(DeepConvNet, self).__init__()
