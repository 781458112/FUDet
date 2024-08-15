import torch
import torch.nn as nn
from abnormal import SelfAttention,softmaxattention1
class Scale(nn.Module):
    def __init__(self, scale=10):
        super(Scale, self).__init__()
        self.scale=nn.Parameter(torch.tensor(scale,dtype=torch.float))

    def forward(self,x):
        return x*self.scale

class AdaptiveNoise_map_Module(nn.Module):
    def __init__(self, input_channel,max_var=0.02):
        super(AdaptiveNoise_map_Module, self).__init__()

        # È«Á¬½Ó²ãÓÃÓÚÉú³ÉÔëÉù¾ùÖµ
        self.fc_mean = nn.Linear(input_channel, 1)

        # È«Á¬½Ó²ãÓÃÓÚÉú³ÉÔëÉù·½²î
        self.fc_var = nn.Linear(input_channel, 1)
        self.max_var = max_var
        # ¼¤»îº¯Êý
        self.relu = nn.ReLU()
        self.attention = SelfAttention(embed_size=1536)
        self.attention_map=softmaxattention1(channel=1536)
        self.mix_noise=1
    def forward(self, x):
        x = self.attention_map(x)
        # È«Á¬½Ó²ãÉú³ÉÔëÉù¾ùÖµºÍ·½²î
        raw_var = self.relu(torch.max(self.fc_var(x)))  # Ê¹ÓÃReLUÈ·±£·½²î·Ç¸º
        var = torch.clamp(raw_var, 0, self.max_var)
        var_cpu=float(var)
        # ¸ù¾Ý¾ùÖµºÍ·½²îÉú³É·ûºÏÕýÌ¬·Ö²¼µÄÔëÉù
        noise = torch.stack([
                        torch.normal(0, var_cpu * 1.1**(k), x.shape)
                        for k in range(self.mix_noise)], dim=1)
        return noise


class AdaptiveNoiseModule(nn.Module):
    def __init__(self, input_dim,max_var=0.02):
        super(AdaptiveNoiseModule, self).__init__()

        # È«Á¬½Ó²ãÓÃÓÚÉú³ÉÔëÉù¾ùÖµ
        self.fc_mean = nn.Linear(input_dim, 1)

        # È«Á¬½Ó²ãÓÃÓÚÉú³ÉÔëÉù·½²î
        self.fc_var = nn.Linear(input_dim, 1)
        self.max_var = max_var
        # ¼¤»îº¯Êý
        self.relu = nn.ReLU()
        self.attention = SelfAttention(embed_size=1536)
        self.mix_noise=1
    def forward(self, x):
        x = self.attention(x)
        # È«Á¬½Ó²ãÉú³ÉÔëÉù¾ùÖµºÍ·½²î
        raw_var = self.relu(torch.max(self.fc_var(x)))  # Ê¹ÓÃReLUÈ·±£·½²î·Ç¸º
        var = torch.clamp(raw_var, 0, self.max_var)
        var_cpu=float(var)
        # ¸ù¾Ý¾ùÖµºÍ·½²îÉú³É·ûºÏÕýÌ¬·Ö²¼µÄÔëÉù
        noise = torch.stack([
                        torch.normal(0, var_cpu * 1.1**(k), x.shape)
                        for k in range(self.mix_noise)], dim=1)
        return noise


class GeneratorWithDynamicNoise(nn.Module):
    def __init__(self, feature_dim=2):
        super(GeneratorWithDynamicNoise, self).__init__()

        # ×ÔÊÊÓ¦ÔëÉùÄ£¿é
        self.noise_module = AdaptiveNoiseModule(feature_dim)
        self.noise_module_featuremap = AdaptiveNoise_map_Module(input_channel=1536)
    def forward(self, x):
        # Éú³ÉÆ÷µÄÇ°Ïò´«²¥²¿·Ö
        # x = x.view(x.size(0), -1)
        # Éú³É·ûºÏÕýÌ¬·Ö²¼µÄÔëÉù
        noise = self.noise_module_featuremap(x)

        # ½«ÔëÉù¼Óµ½Ô­ÌØÕ÷ÏòÁ¿ÉÏ



        return noise