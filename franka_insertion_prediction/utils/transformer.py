import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 正则化
        self.fn = fn  # 具体的操作
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 前向传播
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # attention
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head == dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim = -1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer包含多个编码器的叠加
        for _ in range(depth):
            # 编码器包含两大块：自注意力模块和前向传播模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),  # 多头自注意力模块
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))  # 前向传播模块
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'  # 保证一定能够完整切块
        num_patches = (image_size // patch_size) ** 2  # 获取图像切块的个数
        patch_dim = channels * patch_size ** 2  # 线性变换时的输入大小，即每一个图像宽、高、通道的乘积
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 池化方法必须为cls或者mean

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  # 将批量为b通道为c高为h*p1宽为w*p2的图像转化为批量为b个数为h*w维度为p1*p2*c的图像块
                                                                                                    # 即，把b张c通道的图像分割成b*（h*w）张大小为P1*p2*c的图像块
                                                                                                    # 例如：patch_size为16  (8, 3, 48, 48)->(8, 9, 768)
            nn.Linear(patch_dim, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码，获取一组正态分布的数据用于训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer模块

        self.pool = pool
        self.to_latent = nn.Identity()  # 占位操作

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 正则化
            nn.Linear(dim, num_classes)  # linear output
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        b, n, _ = x.shape  # shape (b, n, 1024)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # cat the token into inputs，x's shape (b, n+1, 1024)
        x += self.pos_embedding[:, :(n + 1)]  # postion encoding，shape (b, n+1, 1024)
        x = self.dropout(x)

        x = self.transformer(x)  # transformer operation

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)  # linear output

class TModel(nn.Module):
    def __init__(self, patch_dim, num_patches, out_dim, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.):
        super().__init__()
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'  # 保证一定能够完整切块
        #num_patches = (image_size // patch_size) ** 2  # 获取图像切块的个数
        #patch_dim = channels * patch_size ** 2  # 线性变换时的输入大小，即每一个图像宽、高、通道的乘积
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 池化方法必须为cls或者mean

        """
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  # 将批量为b通道为c高为h*p1宽为w*p2的图像转化为批量为b个数为h*w维度为p1*p2*c的图像块
                                                                                                    # 即，把b张c通道的图像分割成b*（h*w）张大小为P1*p2*c的图像块
                                                                                                    # 例如：patch_size为16  (8, 3, 48, 48)->(8, 9, 768)
        """
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码，获取一组正态分布的数据用于训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer模块

        self.pool = pool
        self.to_latent = nn.Identity()  # 占位操作

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 正则化
            nn.Linear(dim, out_dim)  # linear output
        )

    def forward(self, x):
        #print(x.type())
        x = self.to_patch_embedding(x)  # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        b, n, _ = x.shape  # shape (b, n, 1024)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # cat the token into inputs，x's shape (b, n+1, 1024)
        x += self.pos_embedding[:, :(n + 1)]  # postion encoding，shape (b, n+1, 1024)
        x = self.dropout(x)

        x = self.transformer(x)  # transformer operation

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)  # linear output


class TransformerLatentModel(nn.Module):
    def __init__(self, patch_dim, num_patches, dim, depth, heads, pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.):
        super().__init__()
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'  # 保证一定能够完整切块
        #num_patches = (image_size // patch_size) ** 2  # 获取图像切块的个数
        #patch_dim = channels * patch_size ** 2  # 线性变换时的输入大小，即每一个图像宽、高、通道的乘积
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 池化方法必须为cls或者mean

        """
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  # 将批量为b通道为c高为h*p1宽为w*p2的图像转化为批量为b个数为h*w维度为p1*p2*c的图像块
                                                                                                    # 即，把b张c通道的图像分割成b*（h*w）张大小为P1*p2*c的图像块
                                                                                                    # 例如：patch_size为16  (8, 3, 48, 48)->(8, 9, 768)
        """
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码，获取一组正态分布的数据用于训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, patch_dim, dropout)  # Transformer模块

        self.pool = pool
        self.to_latent = nn.Identity()  # 占位操作


    def forward(self, x):
        #print(x.type())
        x = self.to_patch_embedding(x)  # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        b, n, _ = x.shape  # shape (b, n, 1024)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # cat the token into inputs，x's shape (b, n+1, 1024)
        x += self.pos_embedding[:, :(n + 1)]  # postion encoding，shape (b, n+1, 1024)
        x = self.dropout(x)

        x = self.transformer(x)  # transformer operation

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return x  # linear output


class TSequenceModel(nn.Module):
    def __init__(self, patch_dim, num_patches, out_dim, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.):
        super().__init__()

        self.seq_model1 = TransformerLatentModel(patch_dim, num_patches, dim, depth//2, heads, pool, dim_head, dropout, emb_dropout)
        self.seq_model2 = TransformerLatentModel(patch_dim, num_patches, dim, depth//2, heads, pool, dim_head, dropout, emb_dropout)
        self.seq_model3 = TransformerLatentModel(patch_dim, num_patches, dim, depth//2, heads, pool, dim_head, dropout, emb_dropout)
        
        self.fusion_model = TransformerLatentModel(dim, 3, dim, depth, heads, pool, dim_head, dropout, emb_dropout)
       
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 正则化
            nn.Linear(dim, out_dim)  # linear output
        )

      
    def forward(self, x):
        seq_emd1 = self.seq_model1(x[:,0])
        seq_emd2 = self.seq_model2(x[:,1])
        seq_emd3 = self.seq_model3(x[:,2])
        #print(seq_emd1.shape,seq_emd2.shape,seq_emd3.shape, )

        seq_emd = torch.stack((seq_emd1,seq_emd2,seq_emd3),dim=1)

        fusion = self.fusion_model(seq_emd)
        
        return self.mlp_head(fusion)  # linear output




class SimpleNet(nn.Module):
    """docstring for SimpleNet"""
    def __init__(self, in_dim=21, out_dim=3, hidden_dim=64):
        super().__init__()

         # encoder takes desired pose + joint configuration
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(int(hidden_dim/2), out_dim)
        )
    
    def forward(self, x):
        out = self.encoder(x)
        return out

    def sample(self, x):
        return self.forward(x)