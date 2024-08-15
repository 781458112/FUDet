import scipy.io as scio
import numpy as np
import random
from torch import tensor,nn
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # Apply linear transformations to obtain queries, keys, and values
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Calculate self-attention scores
        energy = torch.matmul(queries, keys.transpose(1, 0))

        # Normalize with sqrt(d_k), where d_k is the dimension of keys
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** 0.5), dim=1)

        # Apply attention to values
        out = torch.matmul(attention, values)

        # Linear transformation and output
        out = self.fc_out(out)

        return out



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        new_out=torch.cat((x*self.sigmoid(out),x*self.sigmoid(1-out)),dim=1)
        return new_out

class softmaxattention1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,channel):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(softmaxattention1, self).__init__()
        # hidden channels
        #print('c1',c1/8)

        self.cv1 = nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=1,stride=1,padding=0)  # 640*640*9
        self.cv2 = nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=1,stride=1,padding=0)  # 640*640*6
        self.cv3 = nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=1,stride=1,padding=0)
        self.cv4 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.cv5 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.cv6 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.cv7 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.cv8 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.ln=nn.GroupNorm(num_groups=1, num_channels=channel)
        # self.dcn = dcn
        # if self.dcn:
        #     fallback_on_stride = dcn.pop('fallback_on_stride', False)


        #self.cv3 = Conv(6, 3,3, 1,1)  #640*640*3# act=FReLU(c2)
        #Conv(3, 3, 1, 1, 0)
        #self.m 6= nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):


        y1=self.cv1(x)  ##[B,C,H,W]  C
        y2=self.cv2(x)  ##  B
        y3=self.cv3(x)
        ye=self.cv6(x)
        ye_=F.softmax(ye,dim=-1)
        y2_=y2.view([y2.size()[0],y2.size()[1],-1]) #[B,C,H*w] 0 1 2
        #print('y2_', y2_.shape)  ##[B,C,H*w] 0 1 2
        y1_ = y1.view([y1.size()[0], y1.size()[1], -1])  # [B,C,H*w] 0 1 2
        #print('y1_ ', y1_ .shape)  ##[B,C,H*w] 0 1 2
        y2_T = y2_.permute([0,2,1])  # [B,H*w,C]  0 2 1  D   ###K
        #print('y2_T ', y2_T.shape)# [B,H*w,C]
        #c=y1_*y2_T  #[B,C,H*w]  * [B,H*w,C]
        c=torch.matmul(y2_T,y1_) #[B,H*w,H*w]  ###QK
        #print('c ', c.shape)  # [B,H*w,H*w]
        C_weight=F.softmax(c,dim=-1)   ##[B,H*w,H*w]  E
        #print('C_weight', C_weight.shape)  # [B,H*w,H*w]
        #y=C*x
        yee_=self.cv8(self.ln(self.cv7(ye_*y3)))
        y3_=y3.view([x.size()[0], x.size()[1], -1])

        y_last =torch.matmul(y3_,C_weight)

        # y_last = torch.matmul(x_,C_weight) #[B,C,H*w] *[B,H*w,H*w]  F-  ###yuanshi
        y_last_ = y_last.view([x.size()[0], x.size()[1], x.size()[2], x.size()[3]])  # [B,C,H*w] 0 1 2
        # y_last_ = torch.nn.functional.normalize(y_last_)
        # zhao=torch.add(y_last_, x)
        # zhao=torch.cat((y_last_,x),dim=1)
        y_last_a=torch.add(y_last_,yee_)
        avg_x = F.adaptive_avg_pool2d(y_last_a, output_size=1)
        y4 = self.cv4(avg_x)
        y4 = torch.nn.functional.normalize(y4)
        y5 =self.cv5(y4)
        out= y5.reshape(y5.shape[0],y5.shape[1])

        return out




class BidirectionalAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BidirectionalAttentionClassifier, self).__init__()

        # Ç°Ïò×¢ÒâÁ¦µÄÈ¨ÖØ¾ØÕó
        self.W_q_forward = nn.Linear(input_size, hidden_size)
        self.W_k_forward = nn.Linear(input_size, hidden_size)
        self.W_v_forward = nn.Linear(input_size, hidden_size)

        # ·ÖÀàÆ÷µÄÈ¨ÖØ¾ØÕó
        # self.W_out = nn.Linear(hidden_size * 2, num_classes)  # ½áºÏÇ°ÏòºÍ·´Ïò

    def forward(self, x):
        # ¼ÆËãÇ°Ïò×¢ÒâÁ¦
        Q_forward = self.W_q_forward(x)
        K_forward = self.W_k_forward(x)
        V_forward = self.W_v_forward(x)
        attention_scores_forward = F.softmax(torch.matmul(Q_forward, K_forward.t()), dim=-1)
        attention_output_forward = torch.matmul(attention_scores_forward, V_forward)
        attention_output_forward = torch.add(attention_output_forward,x)
        # ¼ÆËã·´Ïò×¢ÒâÁ¦£¬Ê¹ÓÃÈ«Îª1µÄ¾ØÕó¶ÔÇ°Ïò×¢ÒâÁ¦Ïà¼õ
        # attention_scores_backward = 1 - attention_scores_forward
        # attention_output_backward = torch.matmul(attention_scores_backward, V_forward)

        # ½áºÏÇ°ÏòºÍ·´ÏòµÄ×¢ÒâÁ¦Êä³ö
        # attention_output = torch.cat([attention_output_forward, attention_output_backward], dim=-1)

        # Ê¹ÓÃ·ÖÀàÆ÷½øÐÐ×îÖÕ·ÖÀà
        # output = self.W_out(attention_output)

        return attention_output_forward



class abnormal():
    def __init__(self):
        super(abnormal, self).__init__()
    def yichang(self,data):
    ###########Éú³ÉÒì³£Öµ
        # Ìí¼ÓÁ¬Ðø20¸öÒì³£Öµ
        num_groups = 2
        min_pixels_per_group = 4
    # Éú³ÉÒì³£Öµ
        for group in range(num_groups):
            # Ëæ»úÉú³ÉÆðÊ¼ÐÐ¡¢½áÊøÐÐ¡¢ÆðÊ¼ÁÐºÍ½áÊøÁÐ
            start_row = np.random.randint(0, data.shape[0]-min_pixels_per_group)
            end_row = start_row + min_pixels_per_group
            start_col = np.random.randint(0, data.shape[1]-10)
            end_col = start_col + 10
            # Éú³ÉÒì³£Öµ£¬¿ÉÒÔ¸ù¾ÝÐèÒªÐÞ¸ÄÒì³£ÖµµÄÉú³É·½Ê½
            # ÕâÀïÊ¾ÀýÊ¹ÓÃËæ»úÖµ
            outlier_values = np.random.uniform(500, 700, size=(end_row - start_row, end_col - start_col))
            data[start_row:end_row, start_col:end_col] = outlier_values
        return data,(start_row,end_row,start_col,end_col)
    ###########Éú³ÉÈ±Ê§Öµ
    def queshi(self,data,min_missing_pixels = 5):
        num_regions = 2
        for _ in range(num_regions):
            start_row = np.random.randint(0, data.shape[0]-min_missing_pixels)
            end_row = start_row + min_missing_pixels
            start_col = np.random.randint(0, data.shape[1]-10)
            end_col = start_col + 10
            data[start_row:end_row, start_col:end_col] = np.mean(data)
        return data,(start_row,end_row,start_col,end_col)
    #########Éú³ÉÂö³åÕðµ´Öµ
    def zhendang(self,data):
        num_regions = 2  # Éú³É5¸öÇøÓò£¬Äã¿ÉÒÔ¸ù¾ÝÐèÒªµ÷Õû

        # Ëæ»úÉú³ÉÇøÓò²¢¶ÔÃ¿¸öÇøÓò½øÐÐÐÞ¸Ä
        for _ in range(num_regions):
            # Ëæ»úÉú³ÉÇøÓòµÄÆðÊ¼ÐÐ¡¢½áÊøÐÐ¡¢ÆðÊ¼ÁÐºÍ½áÊøÁÐ
            start_row = np.random.randint(0, data.shape[0]-10)
            end_row = start_row + 10
            start_col = np.random.randint(0, data.shape[1]-10)
            end_col = start_col + 10

            # »ñÈ¡ÒªÐÞ¸ÄµÄÊý¾ÝÇøÓò
            region_to_modify = data[start_row:end_row, start_col:end_col]

            # Ëæ»úÑ¡ÔñÔö¼Ó»ò¼õÉÙ²Ù×÷
            if np.random.rand() < 0.5:
                # Ôö¼Ó²Ù×÷
                modification = np.random.uniform(1, 10)  # Ëæ»úÉú³ÉÔö¼ÓµÄÖµ
                modified_region = region_to_modify + modification
            else:
                # ¼õÉÙ²Ù×÷
                modification = np.random.uniform(1, 10)  # Ëæ»úÉú³É¼õÉÙµÄÖµ
                modified_region = region_to_modify - modification

            # ½«ÐÞ¸ÄºóµÄÇøÓò·Å»ØÔ­Ê¼Êý¾ÝÖÐ
            data[start_row:end_row, start_col:end_col] = modified_region
        return data,(start_row,end_row,start_col,end_col)

    def out(self, x):
        a = random.random()
        if a >= 0.3:
            x1, _ = self.queshi(x)
            return x1
        elif 0.3 < a <= 0.6:
            x2, _ = self.yichang(x)
            return x2
        else:
            x3, _ = self.zhendang(x)
            return x3

if __name__ == '__main__':
    a=torch.ones([8,512,224,224])
    model=GlobalContextBlock(in_channels=512)
    b=model(a)
    c=1

