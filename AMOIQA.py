import torch
import torch.nn as nn
from models.swin_transformer_v2 import SwinTransformerV2
import math
import yaml
from functools import partial
import torch.nn.functional as F
from archs.VAR_block import AdaLNSelfAttn, Upsample2x,Downsample2x
from models.swin_transformer_v2 import SwinTransformerV2
import torchvision.models as models




class TDAModule(nn.Module):
    """
        prediction module for top-down attention
    
    """
    def __init__(self, depth=4, embed_dim=192, num_heads=6, mlp_ratio=4., 
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,norm_eps=1e-6, 
                patch_nums=(32, 16, 8, 4), dim_scale = (1, 2, 4, 8),
                attn_l2_norm=True,flash_if_available=True, fused_if_available=True,
                 ):
        super(TDAModule, self).__init__()
        assert embed_dim % num_heads == 0 , f"embed_dim must be divisible by num_heads, but got embed_dim={embed_dim}, num_heads={num_heads}"
        self.depth,self.embed_dim, self.num_heads = depth, embed_dim, num_heads
        self.patch_nums = patch_nums[::-1]
        self.dim_scale = dim_scale[::-1]

        # patch begin/end index
        self.begins_ends = []
        cur = 0
        for pn in self.patch_nums:
            self.begins_ends.append((cur,cur+pn**2))
            cur += pn**2
        self.patch_len = len(self.patch_nums)
        self.L = cur

        # multi-scale feature embedding
        self.feat_embed = nn.ModuleList()
        for scale in self.dim_scale:
            embed = nn.Linear(scale*embed_dim,embed_dim)
            self.feat_embed.append(embed)

        #absolute position embedding
        init_std = math.sqrt(1 / self.embed_dim / 3)
        pos_embed = []
        for pn in self.patch_nums:
            pe = torch.empty(1,pn**2,self.embed_dim)
            nn.init.trunc_normal_(pe,mean = 0,std = init_std)
            pos_embed.append(pe)
        pos_embed = torch.cat(pos_embed,dim=1)

        assert tuple(pos_embed.shape) == (1, self.L, self.embed_dim)
        self.pos_embed = nn.Parameter(pos_embed) #1 L C

        # level embedding and level index
        self.level_embed = nn.Embedding(len(self.patch_nums), self.embed_dim)
        nn.init.trunc_normal_(self.level_embed.weight.data, mean=0, std=init_std)
        level_idx: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)])# L,1
        level_idx = level_idx.view(1, self.L).contiguous()
        self.register_buffer('level_idx', level_idx)# 1,L
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)

        # stochastic depth decay rule (linearly increasing)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        # transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                block_idx=block_idx, embed_dim=self.embed_dim, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        # upsample modules
        self.up = nn.ModuleList()
        for i in range(1,len(self.patch_nums)):
            self.up.append(Upsample2x(self.embed_dim))
 
    
    def forward(self, ms_feat,if_mssal=False):# ms_feat : [B,L,C] L : 32,16,8,4 dim_scale : 1,2,4,8
        ms_feat.reverse()
        assert len(ms_feat) == len(self.dim_scale), f"Input feature length {len(ms_feat)} mismatch"
        embed_feat = []
        # multi-scale feature embedding to the same dimension
        for i,feat in enumerate(ms_feat):
            assert feat.shape[-1] == self.dim_scale[i]*self.embed_dim , f"Feature of layer {i} mismatch,feature channel {feat.shape[-1]}, expecting {self.dim_scale[i]}*{self.embed_dim}."
            feat_embedding = self.feat_embed[i](feat)
            embed_feat.append(feat_embedding) 
        
        # construct the input token map for the first block
        sos = embed_feat[0]
        start = self.begins_ends[0]
        # select the corresponding position embedding and level embedding for the first scale
        lv1_pos = self.pos_embed[:,start[0]:start[1],:]+self.level_embed(self.level_idx[:,start[0]:start[1]])
        # concatenate the input token map with the selected position embedding and level embedding to form the input token map for the first block
        next_token_map =sos+lv1_pos
        # iterate over the remaining blocks and upsample modules to progressively refine the token map 
        for block in self.blocks:
            next_token_map = block(next_token_map)
        sal_maps = [next_token_map.clone()]
        # the first block output is the input token map for the next block
        next_token_map = next_token_map.view(-1,self.patch_nums[0],self.patch_nums[0],self.embed_dim)
        #print(f"feat_embedding {i} shape: {next_token_map.shape}")
        
        # upsample the token map and add the corresponding position embedding and level embedding for each scale,
        for i,pn in enumerate(self.patch_nums[1:]):
            up_token_map = self.up[i](next_token_map.permute(0,3,1,2)).permute(0,2,3,1).contiguous()
            up_token_map = up_token_map.view(-1,pn*pn,self.embed_dim)+embed_feat[i+1]
           
            start,end = self.begins_ends[i+1]
            lv_pos = self.pos_embed[:,start:end,:]+self.level_embed(self.level_idx[:,start:end])
            cur_token_map = up_token_map+lv_pos
            for block in self.blocks:
                cur_token_map = block(cur_token_map)
            sal_maps.append(cur_token_map.clone())
            next_token_map = cur_token_map.view(-1,pn,pn,self.embed_dim)
        sal_maps.reverse() 
        return  sal_maps# 32,16,8,4

class PFAModule(nn.Module):
    """
        Progressive feature aggregation module.
    input: 
        feat : [B,C,h,w] size : 32,16,8,4 dim_scale : 1,2,4,8
        sal_maps : [B,C,h,w] size : 32,16,8,4 dim_scale : 1,2,4,8
    
    """
    def __init__(self, base_ch,patch_nums=(32,16,8,4), dim_scale= (1,2,4,8)):
        super(PFAModule, self).__init__()
        self.patch_nums = patch_nums
        self.dim_scale = dim_scale
        self.base_ch = base_ch
        self.AttnFuse = nn.ModuleList()
        # 1,2,4,8
        fuse_ch = 0 
        for i,dim in enumerate(dim_scale):
            # generate the attention fusion module for each scale
            self.AttnFuse.add_module(f"attn{i}",SpaceAttn(kernelsize=3))
            # adjust the channel of the input feature map for each scale
            self.AttnFuse.add_module(f"channel{i}",nn.Linear(dim*base_ch,base_ch))
            
            if i != len(dim_scale)-1:
                # add up all the channels of the input feature map for each scale
                fuse_ch += base_ch#*self.dim_scale[i]
                # add downsample module for each scale to progressively downsample the feature map 
                self.AttnFuse.add_module(f"down{i+1}",Downsample2x(fuse_ch))
               
    def forward(self, feat, sal_maps):# B L C , B L Ce
        """
            args:
            feat : the input feature maps
            sal_maps : the input weight maps
        """
        assert len(feat) == len(sal_maps), f"Input feature length {len(feat)} mismatch"
        # reverse the order of the input feature maps and weight maps
        feat.reverse() 
        B,_,_ = feat[0].shape
        feat_sal = None
        for i ,(pn, dim) in enumerate(zip(self.patch_nums, self.dim_scale)):
            assert feat[i].shape[2] == dim * self.base_ch, f"{i}expecting {dim}*{self.base_ch}, but got {feat[i].shape[2]}"# BLC
            if feat_sal is not None:
                # use the previous scale's downsample module
                feat_sal = getattr(self.AttnFuse, f"down{i}")(feat_sal)
                x = getattr(self.AttnFuse, f"channel{i}")(feat[i])
                x = x.permute(0,2,1).view(B,self.base_ch,pn,pn).contiguous() 
                x = torch.cat([x,feat_sal],dim=1)
            else:
                x = getattr(self.AttnFuse, f"channel{i}")(feat[i])
                x = x.permute(0,2,1).view(B,self.base_ch,pn,pn).contiguous() # B C H W
            #print(f"{x.shape},{sal_maps[i].shape}")
            attn = getattr(self.AttnFuse, f"attn{i}")(sal_maps[i].permute(0,2,1).view(B,-1,pn,pn).contiguous())
            feat_sal = attn*x # B C H W
        return feat_sal


class SpaceAttn(nn.Module):
    """
        Calculate the attention map under each scale
    """
    def __init__(self, kernelsize = 3,):
        super(SpaceAttn, self).__init__()
        assert kernelsize in [1,3,7], "kernel size should be in [1,3,7]"
        self.conv = nn.Conv2d(2, 1, kernel_size = kernelsize, padding = kernelsize//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        # x: B C H W
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B 1 H W
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return attn

class AMOIQA(nn.Module):
    def __init__(self, backbone="swinv2_tiny_patch4_window8_256",pretrained=False,
                 drop_rate = 0.0, attn_drop_rate=0.0,drop_path_rate=0.0,#CVIQ0.2, 待验证OIQA0.2 IQA-ODI0.0 OIQ 0.0
                 attn_l2_norm=True, flash_if_available=True, fused_if_available=True
                 ):
        super(AMOIQA, self).__init__()
        # load the configuration for the swin backbone
        self.config = yaml.safe_load(open(f'config/{backbone}.yaml','r'))
        self.embed_dim = self.config['MODEL']['SWINV2']['EMBED_DIM']
        self.patch_nums = [self.config['DATA']['IMG_SIZE']//i for i in [8,16,32,64]]
        self.dim_scale = [1,2,4,8]
        self.drop_rate = self.config['MODEL']['DROP_PATH_RATE']
        
        swin = SwinTransformerV2(img_size=self.config['DATA']['IMG_SIZE'],num_classes=0,
                                      embed_dim=self.embed_dim,depths=self.config['MODEL']['SWINV2']['DEPTHS'],
                                      num_heads=self.config['MODEL']['SWINV2']['NUM_HEADS'],
                                      window_size=self.config['MODEL']['SWINV2']['WINDOW_SIZE'],
                                      drop_path_rate=self.drop_rate)
        # load the pretrained weight for the swin backbone, and initialize the weight for the rest of the model
        if pretrained:
            weight = torch.load(f"weights/{backbone}.pth",map_location='cpu')
            weight_dict = None
            for tp in ['model','state_dict']:
                if tp in weight:
                    weight_dict = weight[tp]
            if not weight_dict:
                raise ValueError(f"{tp} not in weight")
            swin_dict = swin.state_dict()
            pretrained_dict = {k: v for k, v in weight_dict.items() if k in swin_dict}
            swin.load_state_dict(pretrained_dict,strict=False)
            print("pretrained weight loaded")
        self.backbone = swin
        # initialize the MultiScaleFuse module
        self.varmodule = TDAModule(depth = 4, embed_dim = 2*self.embed_dim, num_heads = 4, 
                                   patch_nums = self.patch_nums, mlp_ratio = 4, dim_scale = self.dim_scale,
                                   drop_rate = drop_rate, drop_path_rate = drop_path_rate, attn_drop_rate = attn_drop_rate,
                                   attn_l2_norm = attn_l2_norm, flash_if_available = flash_if_available, fused_if_available = fused_if_available)
        # initialize the MultiScaleFuse module
        self.fusemodule = PFAModule(base_ch=2*self.embed_dim, patch_nums=self.patch_nums, dim_scale=self.dim_scale)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        # the regression head for quality prediction
        self.lin = nn.Sequential(nn.Linear(2*self.embed_dim*4*6,4*self.embed_dim),
                                 nn.GELU(),
                                 # IQA-ODI,OIQ设为0.3，CVIQ，OIQA0.5
                                 nn.Dropout(0.3),
                                 nn.Linear(4*self.embed_dim,1))
        # initialize the modules
        self.init_module(self.varmodule)
        self.init_module(self.fusemodule)
        self.init_module(self.lin)
        
        self.init_module(self.channelFuse)
    def init_module(self,module):
        for name, sub_module in module.named_modules():
            if isinstance(sub_module, nn.Linear):
                nn.init.xavier_normal_(sub_module.weight)
                #normal_(sub_module.weight, std=0.02)
                if isinstance(sub_module, nn.Linear) and sub_module.bias is not None:
                    nn.init.constant_(sub_module.bias, 0)
            elif isinstance(sub_module, nn.Conv2d):
                fan_out = sub_module.kernel_size[0] * sub_module.kernel_size[1] * sub_module.out_channels
                fan_out //= sub_module.groups
                nn.init.normal_(sub_module.weight, 0, math.sqrt(2.0 / fan_out))
                if sub_module.bias is not None:
                    nn.init.constant_(sub_module.bias, 0)
    
    def forward(self, img_list):
        # img: B 3 H W
        #print(img_list.shape)
        B,C,H,W = img_list[0].shape
        # check the image size
        assert H==self.config['DATA']['IMG_SIZE'] and W==self.config['DATA']['IMG_SIZE'], f"Image size should be {self.config['DATA']['IMG_SIZE']}"
        feat_list = []
        for img in img_list:
            feat = self.backbone.forward_ms_features(img)
            sal_maps = self.varmodule(feat)
            fuse_feat = self.avg(self.fusemodule(feat,sal_maps))
            feat_list.append(fuse_feat)
        feat_all = torch.cat(feat_list,dim=1)
        out = self.lin(feat_all.reshape(B,-1))
        return out
    def disp(self,list):
        for ls in list:
            print(ls.shape)
