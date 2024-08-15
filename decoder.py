import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 1. ·´¾í»ý²ã
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        # 2. ·´¾í»ý²ã
        self.deconv2 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu2 = nn.ReLU()

        # 3. ·´¾í»ý²ã
        # self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU()

        # 4. ·´¾í»ý²ã
        # self.deconv4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        # # 5. µ÷Õû×îºóÒ»²ã·´¾í»ý²ãµÄ²ÎÊ
        # self.tanh = nn.Tanh()

    def forward(self, x):
        # ÊäÈëxµÄÎ¬¶ÈÎª(8, 6144, 14, 14)

        # Í¨¹ý·´¾í»ý²ãºÍ¼¤»îº¯ÊýÖð²½½âÂë
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        # x = self.relu3(self.bn3(self.deconv3(x)))

        # x = self.tanh(self.deconv4(x))
        # ×îÖÕµÄ·´¾í»ý²ãÊä³ö£¬Ê¹ÓÃtanh¼¤»îº¯Êý½«ÏñËØÖµÏÞÖÆÔÚ[-1, 1]·¶Î§ÄÚ

        # Êä³öxµÄÎ¬¶ÈÎª(8, 3, 224, 224)
        return x
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        # ½«Á½¸öÌØÕ÷ÏòÁ¿Æ´½ÓÔÚÒ»Æð
        x1 = x1.reshape(8,-1)
        x2 = x2.reshape(8,-1)
        return torch.cat((self.layer(x1),self.layer(x2)),dim=0)



