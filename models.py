import torch
import torch.nn as nn
import torch.nn.functional as F


class CompletionNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CompletionNetwork, self).__init__()
        self.__input_shape = input_shape
        self.__img_c = input_shape[0]
        self.__img_h = input_shape[1]
        self.__img_w = input_shape[2]
        # input_shape: (None, img_c, img_h, img_w)
        self.conv1 = nn.Conv2d(self.__img_c, 64, kernel_size=5, stride=1, padding=2)
        # input_shape: (None, 64, img_h, img_w)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # input_shape: (None, 64, img_h, img_w)
        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # input_shape: (None, 32, img_h, img_w)
        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        # output_shape: (None, 3, img_h. img_w)

    def forward(self, x):
        x = F.relu(F.batch_norm(self.conv1(x)))
        x = F.relu(F.batch_norm(self.conv2(x)))
        x = F.relu(F.batch_norm(self.conv3(x)))
        x = F.relu(F.batch_norm(self.conv4(x)))
        x = F.relu(F.batch_norm(self.conv5(x)))
        x = F.relu(F.batch_norm(self.conv6(x)))
        x = F.relu(F.batch_norm(self.conv7(x)))
        x = F.relu(F.batch_norm(self.conv8(x)))
        x = F.relu(F.batch_norm(self.conv9(x)))
        x = F.relu(F.batch_norm(self.conv10(x)))
        x = F.relu(F.batch_norm(self.conv11(x)))
        x = F.relu(F.batch_norm(self.conv12(x)))
        x = F.relu(F.batch_norm(self.deconv13(x)))
        x = F.relu(F.batch_norm(self.conv14(x)))
        x = F.relu(F.batch_norm(self.deconv15(x)))
        x = F.relu(F.batch_norm(self.conv16(x)))
        x = F.sigmoid(self.conv17(x))
        return x

model = CompletionNetwork(input_shape=(3, 256, 256))
print(model)
