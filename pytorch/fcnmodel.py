import torch
from torchvision import models


# read https://arxiv.org/pdf/1505.04366.pdf
# perhaps combine with https://arxiv.org/pdf/1311.2524.pdf
class FCNModel(torch.nn.Module):
    def __init__(self, use_vgg=True):
        super(FCNModel, self).__init__()
        self.use_vgg = use_vgg
        # conv layers
        if not use_vgg:
            # 224x224
            vgg = models.vgg16_bn(pretrained=True)
            self.cnn1_1 = vgg.classifier[0]
            self.cnn1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
            # 112x112
            self.cnn2_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.cnn2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
            # 56x56
            self.cnn3_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
            self.cnn3_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
            self.cnn3_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
            # 28x28
            self.cnn4_1 = torch.nn.Conv2d(256, 512, 3, padding=1)
            self.cnn4_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.cnn4_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
            # 14x14
            self.cnn5_1 = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.cnn5_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.cnn5_3 = torch.nn.Conv2d(512, 512, 3, padding=1)

        else:
            self.vgg = models.vgg16_bn(pretrained=True)
            for param in self.vgg.features.parameters():
                param.require_grad = False

        # 7x7
        self.cnn6 = torch.nn.Conv2d(512, 4096, 7)
        # 1x1
        self.cnn7 = torch.nn.Conv2d(4096, 4096, 1)

        self.out_conv = torch.nn.Conv2d(64, 2, 1)

        # deconv layers
        # 7x7
        self.decnn6 = torch.nn.ConvTranspose2d(4096, 512, 7)

        # 14x14
        self.decnn5_1 = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.decnn5_2 = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.decnn5_3 = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)

        # 28x28
        self.decnn4_1 = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.decnn4_2 = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.decnn4_3 = torch.nn.ConvTranspose2d(512, 256, 3, padding=1)

        # 56x56
        self.decnn3_1 = torch.nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.decnn3_2 = torch.nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.decnn3_3 = torch.nn.ConvTranspose2d(256, 128, 3, padding=1)

        # 112x112
        self.decnn2_1 = torch.nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.decnn2_2 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)

        # 224x224
        self.decnn1_1 = torch.nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.decnn1_2 = torch.nn.ConvTranspose2d(64, 64, 3, padding=1)

        # pooling layers
        self.pool2d = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2d = torch.nn.MaxUnpool2d(2, stride=2)

        # activation layers
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax2d()

        # other layers
        self.dropout = torch.nn.Dropout2d()
        self.batchnorm1_1 = torch.nn.BatchNorm2d(64)
        self.batchnorm1_2 = torch.nn.BatchNorm2d(64)
        self.batchnorm2_1 = torch.nn.BatchNorm2d(128)
        self.batchnorm2_2 = torch.nn.BatchNorm2d(128)
        self.batchnorm3_1 = torch.nn.BatchNorm2d(256)
        self.batchnorm3_2 = torch.nn.BatchNorm2d(256)
        self.batchnorm3_3 = torch.nn.BatchNorm2d(256)
        self.batchnorm4_1 = torch.nn.BatchNorm2d(512)
        self.batchnorm4_2 = torch.nn.BatchNorm2d(512)
        self.batchnorm4_3 = torch.nn.BatchNorm2d(512)
        self.batchnorm5_1 = torch.nn.BatchNorm2d(512)
        self.batchnorm5_2 = torch.nn.BatchNorm2d(512)
        self.batchnorm5_3 = torch.nn.BatchNorm2d(512)
        self.batchnorm6 = torch.nn.BatchNorm2d(4096)

        self.up_batchnorm6 = torch.nn.BatchNorm2d(512)
        self.up_batchnorm5_1 = torch.nn.BatchNorm2d(512)
        self.up_batchnorm5_2 = torch.nn.BatchNorm2d(512)
        self.up_batchnorm5_3 = torch.nn.BatchNorm2d(512)
        self.up_batchnorm4_1 = torch.nn.BatchNorm2d(512)
        self.up_batchnorm4_2 = torch.nn.BatchNorm2d(512)
        self.up_batchnorm4_3 = torch.nn.BatchNorm2d(256)
        self.up_batchnorm3_1 = torch.nn.BatchNorm2d(256)
        self.up_batchnorm3_2 = torch.nn.BatchNorm2d(256)
        self.up_batchnorm3_3 = torch.nn.BatchNorm2d(128)
        self.up_batchnorm2_1 = torch.nn.BatchNorm2d(128)
        self.up_batchnorm2_2 = torch.nn.BatchNorm2d(64)
        self.up_batchnorm1_1 = torch.nn.BatchNorm2d(64)
        self.up_batchnorm1_2 = torch.nn.BatchNorm2d(64)

    def forward(self, x):

        if self.use_vgg:
            features = list(self.vgg.features.children())
            result = features[0](x)
            result = features[1](result)
            result = features[2](result)
            result = features[3](result)
            result = features[4](result)
            result = features[5](result)

            mpool = features[6]
            mpool.return_indices = True
            result, indices1 = mpool(result)

            result = features[7](result)
            result = features[8](result)
            result = features[9](result)
            result = features[10](result)
            result = features[11](result)
            result = features[12](result)

            mpool = features[13]
            mpool.return_indices = True
            result, indices2 = mpool(result)

            result = features[14](result)
            result = features[15](result)
            result = features[16](result)
            result = features[17](result)
            result = features[18](result)
            result = features[19](result)
            result = features[20](result)
            result = features[21](result)
            result = features[22](result)

            mpool = features[23]
            mpool.return_indices = True
            result, indices3 = mpool(result)

            result = features[24](result)
            result = features[25](result)
            result = features[26](result)
            result = features[27](result)
            result = features[28](result)
            result = features[29](result)
            result = features[30](result)
            result = features[31](result)
            result = features[32](result)

            mpool = features[33]
            mpool.return_indices = True
            result, indices4 = mpool(result)

            result = features[34](result)
            result = features[35](result)
            result = features[36](result)
            result = features[37](result)
            result = features[38](result)
            result = features[39](result)
            result = features[40](result)
            result = features[41](result)
            result = features[42](result)

            mpool = features[43]
            mpool.return_indices = True
            result, indices5 = mpool(result)
        else:
            result = self.cnn1_1(x)
            result = self.batchnorm1_1(result)
            result = self.relu(result)
            result = self.cnn1_2(result)
            result = self.batchnorm1_2(result)
            result = self.relu(result)
            result, indices1 = self.pool2d(result)

            result = self.cnn2_1(result)
            result = self.batchnorm2_1(result)
            result = self.relu(result)
            result = self.cnn2_2(result)
            result = self.batchnorm2_2(result)
            result = self.relu(result)
            result, indices2 = self.pool2d(result)

            result = self.cnn3_1(result)
            result = self.batchnorm3_1(result)
            result = self.relu(result)
            result = self.cnn3_2(result)
            result = self.batchnorm3_2(result)
            result = self.relu(result)
            result = self.cnn3_3(result)
            result = self.batchnorm3_3(result)
            result = self.relu(result)
            result, indices3 = self.pool2d(result)

            result = self.cnn4_1(result)
            result = self.batchnorm4_1(result)
            result = self.relu(result)
            result = self.cnn4_2(result)
            result = self.batchnorm4_2(result)
            result = self.relu(result)
            result = self.cnn4_3(result)
            result = self.batchnorm4_3(result)
            result = self.relu(result)
            result, indices4 = self.pool2d(result)

            result = self.cnn5_1(result)
            result = self.batchnorm5_1(result)
            result = self.relu(result)
            result = self.cnn5_2(result)
            result = self.batchnorm5_2(result)
            result = self.relu(result)
            result = self.cnn5_3(result)
            result = self.batchnorm5_3(result)
            result = self.relu(result)
            result, indices5 = self.pool2d(result)

        result = self.cnn6(result)
        result = self.batchnorm6(result)
        result = self.relu(result)

        result = self.decnn6(result)
        result = self.up_batchnorm6(result)
        result = self.relu(result)

        result = self.unpool2d(result, indices5)

        result = self.decnn5_1(result)
        result = self.up_batchnorm5_1(result)
        result = self.relu(result)
        result = self.decnn5_2(result)
        result = self.up_batchnorm5_2(result)
        result = self.relu(result)
        result = self.decnn5_3(result)
        result = self.up_batchnorm5_3(result)
        result = self.relu(result)

        result = self.unpool2d(result, indices4)

        result = self.decnn4_1(result)
        result = self.up_batchnorm4_1(result)
        result = self.relu(result)
        result = self.decnn4_2(result)
        result = self.up_batchnorm4_2(result)
        result = self.relu(result)
        result = self.decnn4_3(result)
        result = self.up_batchnorm4_3(result)
        result = self.relu(result)

        result = self.unpool2d(result, indices3)

        result = self.decnn3_1(result)
        result = self.up_batchnorm3_1(result)
        result = self.relu(result)
        result = self.decnn3_2(result)
        result = self.up_batchnorm3_2(result)
        result = self.relu(result)
        result = self.decnn3_3(result)
        result = self.up_batchnorm3_3(result)
        result = self.relu(result)

        result = self.unpool2d(result, indices2)

        result = self.decnn2_1(result)
        result = self.up_batchnorm2_1(result)
        result = self.relu(result)
        result = self.decnn2_2(result)
        result = self.up_batchnorm2_2(result)
        result = self.relu(result)

        result = self.unpool2d(result, indices1)

        result = self.decnn1_1(result)
        result = self.up_batchnorm1_1(result)
        result = self.relu(result)
        result = self.decnn1_2(result)
        result = self.up_batchnorm1_2(result)
        result = self.relu(result)
        result = self.out_conv(result)
        # result = self.softmax(result)
        # result = self.sigmoid(result)
        return result
