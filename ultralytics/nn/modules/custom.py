import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DySample', 'ResEMA']

class DySample(nn.Module):
    def __init__(self, c1, *args, **kwargs):
        super().__init__()
        self.scale = 2
        self.style = 'lp'
        self.groups = 4
        
        # Robust Argument Parsing: Scan *args for scale (int) and style (str)
        # This prevents errors if YOLO shifts arguments.
        for arg in args:
            if isinstance(arg, int): 
                self.scale = arg
            elif isinstance(arg, str): 
                self.style = arg
            
        if self.style == 'pl':
            offset_in = c1 // (self.scale ** 2)
            out_channels = 2 * self.groups
        else: 
            offset_in = c1
            out_channels = 2 * self.groups * self.scale ** 2
            
        self.offset = nn.Conv2d(offset_in, out_channels, 1, bias=False)
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear', align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

class ResEMA(nn.Module):
    def __init__(self, c1, *args, **kwargs):
        super().__init__()
        self.groups = 8
        
        # Robust Argument Parsing
        for arg in args:
            if isinstance(arg, int) and arg < 100: 
                self.groups = arg
        
        # Safety check for groups
        if c1 // self.groups <= 0: 
            self.groups = 1

        mid_channels = c1 // 2 if c1 > 1 else c1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(c1, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(mid_channels, c1, 1, bias=False),
            nn.BatchNorm2d(c1)
        )
        self.internal_c = c1 // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(min(self.internal_c, 32), self.internal_c)
        self.conv1x1 = nn.Conv2d(self.internal_c, self.internal_c, 1, bias=False)
        self.conv3x3 = nn.Conv2d(self.internal_c, self.internal_c, 3, padding=1, bias=False)

    def forward(self, x):
        residual = x 
        y = self.conv_block1(x)
        y = self.conv_block2(y)
        b, c, h, w = y.size()
        group_x = y.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        
        # --- FIXED WEIGHT CALCULATION ---
        # 1. Calculate weights. Shape comes out as (B * Groups, 1, Channels)
        w1 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        w2 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        
        # 2. FIX: Reshape to (B * Groups, Channels, 1, 1) so it can broadcast correctly over (H, W)
        w1 = w1.permute(0, 2, 1).reshape(b * self.groups, self.internal_c, 1, 1)
        w2 = w2.permute(0, 2, 1).reshape(b * self.groups, self.internal_c, 1, 1)
        # --------------------------------

        out = (x1 * w1 + x2 * w2).reshape(b, c, h, w)
        return residual + out