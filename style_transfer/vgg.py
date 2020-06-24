import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), vgg[x])

    def forward(self, x):
        out1 = self.to_relu_1_2(x)
        out2 = self.to_relu_2_2(out1)
        out3 = self.to_relu_3_3(out2)
        out4 = self.to_relu_4_3(out3)
        return (out1, out2, out3, out4)
