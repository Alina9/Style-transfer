from PIL.Image import Image
import PIL
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from Net import Style_transfer_net
from utils import gram, transform
from vgg import VGG


class Style_transfer:
    def __init__(self, style, epoch, style_weight, content_weight, tv_weight, batch_size,
                 path="val2017", load=False):
        self.style = Variable(style.repeat(batch_size, 1, 1, 1)).type(torch.FloatTensor)
        self.epoch = epoch
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.vgg = VGG()
        if load:
            self.load()
        else:
            self.style_net = Style_transfer_net()
        self.optimizer = optim.Adam(self.style_net.parameters(), 1e-3)
        self.train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(path, transform=transform),
                                                        batch_size=batch_size,
                                                        num_workers=0,
                                                        shuffle=True)

        self.mse_loss = nn.MSELoss()

    def load(self):
        self.style_net = Style_transfer_net()
        self.style_net.load_state_dict(torch.load("models/style_net.pkl"))

    def save(self):
        torch.save(self.style_net.state_dict(), "models/style_mosaic.pkl")

    def train(self):
        features_style = self.vgg(self.style)
        gram_style = [gram(y.detach()) for y in features_style]

        for e in range(self.epoch):
            i = 0
            for x, _ in iter(self.train_loader):
                i += 1
                batch_size = len(x)
                x = Variable(x)
                y = self.style_net(x)

                features_y = self.vgg(y)
                features_x = self.vgg(x)
                content_loss = self.content_weight * F.mse_loss(features_y[2], features_x[2])

                y_hat_gram = [gram(fmap) for fmap in features_y]
                style_loss_local = 0.0

                for j in range(4):
                    style_loss_local += F.mse_loss(y_hat_gram[j], gram_style[j][:batch_size])
                style_loss = self.style_weight * style_loss_local

                diff_i = torch.sum(torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :]))
                tv_loss = self.tv_weight * (diff_i + diff_j)

                total_loss = style_loss + content_loss + tv_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                if i % 100 == 0 or i == 1:
                    print(f"Epoch: {e}, iter: {i}, loss: {total_loss.item()}, "
                          f"style: {style_loss.item()},content: {content_loss.item()}, tv: {tv_loss}")
                    print("SAVED")
                    self.save()

            print("SAVED")
            self.save()



if __name__ == "__main__":
    style = PIL.Image.open('style_image/mosaic.jpg')
    style = transform(style)
    style = torch.unsqueeze(style, 0)
    transfer = Style_transfer(style, epoch=2, style_weight=1e1, content_weight=1e0, tv_weight=1e-7, batch_size=4,
                              load=True)
    transfer.train()