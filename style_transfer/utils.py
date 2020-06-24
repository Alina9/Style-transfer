import torch
from torchvision import transforms

img_size = 512

transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.CenterCrop(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255))])

transform_bw = transforms.Compose([transforms.Resize(img_size),
                                transforms.CenterCrop(img_size),
                           transforms.ToTensor(),
                            transforms.Lambda(lambda x: torch.cat([x,x,x])),
                           transforms.Lambda(lambda x: x.mul_(255))])

transform_for_show = transforms.Compose([transforms.Lambda(lambda x: x.clone()[0]),
                           transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                           transforms.Lambda(lambda x: x.data.clamp(0, 1)),
                           transforms.ToPILImage()
                           ])

transform_for_show_bw = transforms.Compose([transforms.Lambda(lambda x: x.clone()[0]),
                           transforms.Lambda(lambda x: x.mul_(1./255)),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                           transforms.Lambda(lambda x: x.data.clamp(0, 1)),
                           transforms.ToPILImage()
                           ])


def gram(x):
    Nx, Cx, Hx, Wx = x.size()
    features = x.view(Nx, Cx, Hx * Wx)
    G = torch.bmm(features, features.transpose(1, 2))
    G = G.div(Cx * Hx * Wx)
    return G
