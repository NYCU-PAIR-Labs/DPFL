import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models


class cnn_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(cnn_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.main(x)

# 5-layer CNN
class MyNet(nn.Module):
    def __init__(self, dim=16, num_classes=10):
        super(MyNet, self).__init__()
        self.cnn_block1 = cnn_layer(3, dim*2)
        self.cnn_block2 = cnn_layer(dim*2, dim*2)
        self.cnn_block3 = cnn_layer(dim*2, dim*4)
        self.cnn_block4 = cnn_layer(dim*4, dim*8)
        self.cnn_block5 = cnn_layer(dim*8, dim*8)
        self.fc = nn.Sequential(
            nn.Linear(dim*8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mp = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(self.mp(x))
        x = self.cnn_block3(self.mp(x))
        x = self.cnn_block4(self.mp(x))
        x = self.cnn_block5(self.mp(x))

        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        x = self.fc(feature)
        return x, feature
    
# 2-layer CNN
class MyNet_Simple(nn.Module):
    def __init__(self, dim=16, num_classes=10):
        super(MyNet_Simple, self).__init__()
        self.cnn_block1 = cnn_layer(3, dim*2)
        self.cnn_block2 = cnn_layer(dim*2, dim*4)
        self.bottle_neck = nn.Linear(dim*4*8*8, dim*4)

        self.fc = nn.Sequential(
            nn.Linear(dim*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        self.mp = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.mp(x)
        x = self.cnn_block2(x)
        x = self.mp(x)

        feature = self.bottle_neck(x.view(x.size(0), -1))
        x = self.fc(feature)
        return x, feature
    
    def feature_extractor(self, x):
        x = self.cnn_block1(x)
        x = self.mp(x)
        x = self.cnn_block2(x)
        x = self.mp(x)

        feature = self.bottle_neck(x.view(x.size(0), -1))
        return feature
    
    def classifier(self, feature):
        return self.fc(feature)



''' CNN from NIID-Bench '''
class NIID_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(NIID_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        feature = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, feature


''' Conditional Generator for capturing the data distribution of clients '''
class CGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, img_size=32, n_cls=10):
        super(CGenerator, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
        self.l2 = nn.Sequential(nn.Linear(n_cls, ngf*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.up = nn.functional.interpolate(scale_factor=2)

    def forward(self, z, y):
        out_1 = self.l1(z.view(z.shape[0],-1))
        out_2 = self.l2(y.view(y.shape[0],-1))
        out = torch.cat([out_1, out_2], dim=1)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = self.normalize(img)
        return img
    

class FedGen_Generator(nn.Module):
    def __init__(self, z_dim=32, output_dim=64, num_classes=10):
        super(FedGen_Generator, self).__init__()

        self.z_dim = z_dim
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.generator = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh(),
            # nn.BatchNorm1d(output_dim),
        )

    def forward(self, z, label):
        onehot_label = F.one_hot(label, num_classes=self.num_classes).to(label.device)     # convert to one-hot label
        _input = torch.cat((z, onehot_label), dim=1)

        output = self.generator(_input)
        return output


class DCGAN(nn.Module):
    def __init__(self, z_dim=100, hidden_dim=64, output_dim=3, num_classes=10):
        super(DCGAN, self).__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(z_dim + num_classes, hidden_dim*4)

        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(hidden_dim * 2, output_dim, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(ngf) x 32 x 32``
        )

    def forward(self, x, label):
        label = F.one_hot(label, num_classes=self.num_classes).to(x.device)
        z = torch.cat((x, label), dim=1)

        output = self.fc1(z)
        output = output.view(-1, output.shape[1], 1, 1)
        output = self.generator(output)
        return output