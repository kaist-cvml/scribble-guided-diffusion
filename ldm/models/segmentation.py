import torch
from torch import nn
from torch.nn import SiLU
import math


class DownScaleLayer(nn.Module):
    def __init__(self, in_c, out_c, mode, factor=2):
        super(DownScaleLayer, self).__init__()
        if mode=="avgpool":
            self.downscale = nn.Sequential(nn.AvgPool2d(factor),
                                           nn.Conv2d(in_c,out_c,1))
        elif mode=="maxpool":    
            self.downscale = nn.Sequential(nn.MaxPool2d(factor),
                                           nn.Conv2d(in_c,out_c,1))
        elif mode=="conv":
            self.downscale = nn.Conv2d(in_c,out_c,factor,factor)
        else:
            raise ValueError("did not recognize mode: "+mode)

    def forward(self, x):
        return self.downscale(x)


class UpScaleLayer(nn.Module):
    def __init__(self, in_c, out_c, mode, factor=2):
        super(UpScaleLayer, self).__init__()
        if mode in ['nearest','linear','bilinear','bicubic','trilinear']:
            self.upscale = nn.Sequential(nn.Conv2d(in_c,out_c,1),
                                         nn.Upsample(scale_factor=factor,mode=mode,align_corners=False))
        elif mode=="tconv":    
            self.upscale = nn.ConvTranspose2d(in_c,out_c,factor,factor)
        else:
            raise ValueError("did not recognize mode: "+mode)

    def forward(self, x):
        return self.upscale(x)


class SELayer(nn.Module):
    def __init__(self, in_c, hidden_c, reduction=4,act=SiLU()):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(in_c, hidden_c),
                act,
                nn.Linear(hidden_c, in_c),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

"""
def conv_layer(in_c,out_c,k=1,bias=False,use_bn=False,act=nn.ReLU(),stride=1):
    return nn.Sequential(*(
        [nn.Conv2d(in_c, out_c, k, stride, k//2, bias=bias)]+
        ([nn.BatchNorm2d(out_c)] if use_bn else [])+
        ([act] if act is not None else [])
        ))"""

def get_act(act_name="relu"):
    if act_name.lower()=="relu":
        act = nn.ReLU()
    elif act_name.lower()=="silu":
        act = SiLU()
    else:
        raise ValueError("Did not recognize activation: "+act_name)
    return act


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, expand_ratio=4, use_se=False, use_bn=False, act=nn.ReLU(), stride=1):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_c * expand_ratio)
        self.identity = stride == 1 and in_c == out_c
        self.conv = nn.Sequential(*(
            [nn.Conv2d(in_c, hidden_dim, 1, 1, 0, bias=False)]+
            ([nn.BatchNorm2d(hidden_dim)] if use_bn else [])+
            [act]+
            
            [nn.Conv2d(hidden_dim, hidden_dim, k, stride, k//2, groups=hidden_dim, bias=False)]+
            ([nn.BatchNorm2d(hidden_dim)] if use_bn else [])+
            [act]+
            ([SELayer(hidden_dim, in_c//4)] if use_se else [])+
            
            [nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False)]+
            ([nn.BatchNorm2d(out_c)] if use_bn else [])
        ))
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FusedMBConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, expand_ratio=4, use_se=False, use_bn=False, act=nn.ReLU(), stride=1):
        super(FusedMBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_c * expand_ratio)
        self.identity = stride == 1 and in_c == out_c
        
        self.conv = nn.Sequential(*(
                [nn.Conv2d(in_c, hidden_dim, k, stride, k//2, bias=False)]+
                ([nn.BatchNorm2d(hidden_dim)] if use_bn else [])+
                [act]+
                ([SELayer(hidden_dim, in_c//4)] if use_se else [])+
                
                [nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False)]+
                ([nn.BatchNorm2d(out_c)] if use_bn else [])
            ))
    def forward(self, x):
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)


BLOCK_DICT = {"m": MBConv,
              "f": FusedMBConv,
              "u": "UNetConv"}


class AbstractUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        assert len(args.unet.num_c) == len(args.unet.num_repeat)
        assert len(args.unet.num_c) == len(args.unet.expand_ratio)
        assert len(args.unet.num_c) == len(args.unet.SE)
        
        config = zip(range(args.unet.num_blocks),
                     args.unet.block,
                     args.unet.num_c,
                     args.unet.num_repeat,
                     args.unet.expand_ratio,
                     args.unet.SE)
        
        act = get_act(act_name=args.unet.act)
        
        self.first_conv = nn.Conv2d(args.unet.input_channels, args.unet.num_c[0], 3, 1, 1, bias=False)
        self.last_conv = nn.Conv2d(args.unet.num_c[0], 1, 3, 1, 1, bias=False)
        #self.first_conv = conv_layer(args.unet.input_channels, args.unet.num_c[0], 3, 1, 1)
        #self.last_conv = conv_layer(args.unet.num_c[0], 1, 3, 1, 1)
        
        
        DownBlocks = []
        UpBlocks = []
        
        downscales = []
        upscales = []
        
        channels_prev = -1
        
        for i, block_type, channels, num_rep, ratio, use_se in config:
            if channels_prev>0:
                downscales.append(DownScaleLayer(channels_prev,
                                                 channels,
                                                 mode=args.unet.downscale_mode))
                upscales.append(UpScaleLayer(channels,
                                             channels_prev,
                                             mode=args.unet.upscale_mode))
        
            DownBlocks.append(nn.Sequential(*[BLOCK_DICT[block_type](channels, 
                                            channels, 
                                            k=3, 
                                            expand_ratio=ratio, 
                                            use_se=use_se, 
                                            use_bn=True, 
                                            act=act, 
                                            stride=1) for _ in range(num_rep)]))
            if args.unet.res_mode=="cat":
                channels_r0 = channels*2 if i+1 < args.unet.num_blocks else channels
            else:
                channels_r0 = channels
            
            UpBlocks.append(nn.Sequential(*[BLOCK_DICT[block_type](channels_r0 if r==0 else channels, 
                                            channels, 
                                            k=3, 
                                            expand_ratio=ratio, 
                                            use_se=use_se, 
                                            use_bn=True, 
                                            act=act, 
                                            stride=1) for r in range(num_rep)]))
            
            channels_prev = channels
        
        self.DownBlocks = nn.ModuleList(DownBlocks)
        self.UpBlocks = nn.ModuleList(UpBlocks)
        
        self.downscales = nn.ModuleList(downscales)
        self.upscales = nn.ModuleList(upscales)
        
        
        if args.unet.init_mode=="effecientnetv2":
            self._initialize_weights_effecientnetv2()

    def _initialize_weights_effecientnetv2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        skip = []
        
        x = self.first_conv(x)
        for i in range(self.args.unet.num_blocks):
            x = self.DownBlocks[i](x)
            
            if i+1 < self.args.unet.num_blocks:
                skip.append(x)
                x = self.downscales[i](x)
                
        for i in reversed(range(self.args.unet.num_blocks)):
            if i+1 < self.args.unet.num_blocks:
                x = self.upscales[i](x)
                if self.args.unet.res_mode=="cat":
                    x = torch.cat((x,skip.pop()),dim=1)
                elif self.args.unet.res_mode=="add":
                    x += skip.pop()
                
            x = self.UpBlocks[i](x)
            
        return self.last_conv(x)


class AbstractDownNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        assert len(args.adv.num_c) == len(args.adv.num_repeat)
        assert len(args.adv.num_c) == len(args.adv.expand_ratio)
        assert len(args.adv.num_c) == len(args.adv.SE)
        
        config = zip(range(args.adv.num_blocks),
                     args.adv.block,
                     args.adv.num_c,
                     args.adv.num_repeat,
                     args.adv.expand_ratio,
                     args.adv.SE)
        
        act = get_act(act_name=args.adv.act)
        
        #self.first_conv = conv_layer(args.adv.input_channels, args.adv.num_c[0], 3, 1, 1)
        self.first_conv = nn.Conv2d(args.adv.input_channels, args.adv.num_c[0], 3, 1, 1, bias=False)
        self.n_features = args.adv.num_c[-1]*args.adv.reshape_last[0]*args.adv.reshape_last[1]
        
        
        
        if len(args.adv.fc_c)>0:
            fc = [nn.Linear(self.n_features,args.adv.fc_c[0])]
            self.adaptive_pool = nn.AdaptiveAvgPool2d(args.adv.reshape_last)
        else:
            fc = [nn.Conv2d(args.adv.num_c[-1],1,1)]
            
        if len(args.adv.fc_c)>1:
            for i in range(len(args.adv.fc_c)-1):
                fc.append(act)
                fc.append(nn.Linear(args.adv.fc_c[i],args.adv.fc_c[i+1]))
        
        self.FC = nn.Sequential(*fc)
        
        DownBlocks = []
        
        downscales = []
        
        channels_prev = -1
        
        for i, block_type, channels, num_rep, ratio, use_se in config:
            if channels_prev>0:
                downscales.append(DownScaleLayer(channels_prev,
                                                 channels,
                                                 mode=args.adv.downscale_mode))
        
            DownBlocks.append(nn.Sequential(*[BLOCK_DICT[block_type](channels, 
                                            channels, 
                                            k=3, 
                                            expand_ratio=ratio, 
                                            use_se=use_se, 
                                            use_bn=True, 
                                            act=act, 
                                            stride=1) for _ in range(num_rep)]))
            
            channels_prev = channels
        
        self.DownBlocks = nn.ModuleList(DownBlocks)
        
        self.downscales = nn.ModuleList(downscales)
        
        if args.adv.init_mode=="effecientnetv2":
            self._initialize_weights_effecientnetv2()

    def _initialize_weights_effecientnetv2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        skip = []
        
        x = self.first_conv(x)
        for i in range(self.args.adv.num_blocks):
            x = self.DownBlocks[i](x)
            
            if i+1 < self.args.adv.num_blocks:
                skip.append(x)
                x = self.downscales[i](x)
                
        if len(self.args.adv.fc_c)>0:
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            x = self.FC(x)
        else:
            x = self.FC(x)
            x = x.mean()
        
        return x