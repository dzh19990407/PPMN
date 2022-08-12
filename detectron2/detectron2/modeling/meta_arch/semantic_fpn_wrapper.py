import torch
import torch.nn as nn
from .positional_encoding import build_positional_encoding


class SemanticFPNWrapper(nn.Module):

    def __init__(self):
        super(SemanticFPNWrapper, self).__init__()
        self.cat_coors_level = 3
        ######
        self.convs_all_levels = nn.ModuleList()
        conv_lay0 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        self.convs_all_levels.append(conv_lay0)
        conv_lay1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        self.convs_all_levels.append(conv_lay1)
        conv_lay2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2.0),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        self.convs_all_levels.append(conv_lay2)
        conv_lay3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2.0),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2.0),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        self.convs_all_levels.append(conv_lay3)
        ######
        self.conv_pred = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        ######
        self.aux_convs = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=1),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m, std=0.01)

    def forward(self, inputs): #inputs: extract_features
        # inputs: [p2, p3, p4, p5]
        import pdb; pdb.set_trace()
        mlvl_feats = []
        for i in range(4):
            input_p = inputs[i]
            if i == self.cat_coors_level: 
                ignore_mask = input_p.new_zeros(
                    (input_p.shape[0], input_p.shape[-2],
                     input_p.shape[-1]),
                     dtype=torch.bool) #ignore_mask.shape=[2,25,38]
                positional_encoding = build_positional_encoding(ignore_mask)
                input_p = input_p + positional_encoding

            mlvl_feats.append(self.convs_all_levels[i](input_p))
        import pdb; pdb.set_trace()

        feature_add_all_level = sum(mlvl_feats)

        out = self.conv_pred(feature_add_all_level) #[2,256,100,152]

        outs = [out]
        outs.append(self.aux_convs(feature_add_all_level)) #[2,256,100,152]
        import pdb; pdb.set_trace()
            # outs: list, 2
            # outs[0]: [B, 256, H//8, W//8]
            # outs[1]: [B, 256, H//8, W//8]
        return outs
