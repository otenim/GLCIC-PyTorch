import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Flatten, Concatenate


class CompletionNetwork(nn.Module):

    def __init__(self):
        super(CompletionNetwork, self).__init__()
        # input_shape: (None, 3, img_h, img_w)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()
        # output_shape: (None, 3, img_h. img_w)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.act17(self.conv17(x))
        return x


class LocalDiscriminator(nn.Module):

    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]
        # input_shape: (None, img_c, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()
        # input_shape: (None, 256, img_h//8, img_w//8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()
        # input_shape: (None, 512, img_h//16, img_w//16)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten6 = Flatten()
        # input_shape: (None, 512 * img_h//32 * img_w//32)
        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()
        # output_shape: (None, 1024)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        return x


class GlobalDiscriminator(nn.Module):

    def __init__(self, input_shape):
        super(GlobalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        # input_shape: (None, img_c, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()
        # input_shape: (None, 256, img_h//8, img_w//8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()
        # input_shape: (None, 512, img_h//16, img_w//16)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten6 = Flatten()
        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()
        # output_shape: (None, 1024)


    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        return x


class ContextDiscriminator(nn.Module):

    def __init__(self, local_input_shape, global_input_shape):

        super(ContextDiscriminator, self).__init__()
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)
        self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_gd = GlobalDiscriminator(global_input_shape)
        # input_shape: [(None, 1024), (None, 1024)]
        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        self.concat1 = Concatenate(dim=-1)
        # input_shape: (None, 2048)
        self.linear1 = nn.Linear(in_features, 1)
        self.act1 = nn.Sigmoid()
        # output_shape: (None, 1)

    def forward(self, x):
        x_ld, x_gd = x
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        out = self.act1(self.linear1(self.concat1([x_ld, x_gd])))
        return out
