# -*- coding: utf-8 -*-

"""
Panoptic Narrative Grounding Baseline Network PyTorch implementation.
"""

import torch
import torch.nn as nn

from .encoder_bert import BertEncoder

class PanopticNarrativeGroundingBaseline(nn.Module):
    def __init__(self, cfg,
                 device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Define the network
        self.bert_encoder = BertEncoder(
            cfg,
        )
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, sent, pos, feat, noun_phrases):
        """
        :param feat: b, 2, o, f     feat.shape=[60,58,50176]    #mask
        :param pos:  b, 2, o, 4     pos.shape=[60,58,4]     #box
        :param sent: b, (string)    len(list)=60    #captions
        :param noun_phrases: b, l, np   noun_phrases.shape=[60,230,28]
        :return:
        """
        output_lang, output_img, _ = self.bert_encoder(sent, (feat, pos), noun_phrases) #output_lang.shape=[60,230,768]
        output_img = output_img.permute([0, 2, 1]) #[60,768,58]
        output = torch.matmul(output_lang, output_img) #[60,230,58]

        return output
