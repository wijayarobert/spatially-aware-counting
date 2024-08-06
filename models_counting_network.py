from functools import partial
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed
import torchvision.transforms as transforms 
import numpy as np
from timm.models.vision_transformer import Block

class CountingNetwork(nn.Module):
    def __init__(
        self,
        img_encoder_num_output_tokens=256,
        fim_embed_dim=768,
        fim_depth=2,
        fim_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):

    # def __init__(self, img_size=384, patch_size=16, in_chans=3,
    #     embed_dim=1024, depth=24, num_heads=16,
    #     decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
    #     mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):    
        
        super().__init__()

        # --------------------------------------------------------------------------
        # Feature interaction module specifics.
        self.fim_num_img_tokens = img_encoder_num_output_tokens

        # Use a fixed sin-cos embedding.
        self.fim_pos_embed = nn.Parameter(
            torch.zeros(1, self.fim_num_img_tokens, fim_embed_dim), requires_grad=False
        )

        self.fim_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    fim_embed_dim,
                    fim_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(fim_depth)
            ]
        )

        self.fim_norm = norm_layer(fim_embed_dim)

        # --------------------------------------------------------------------------
        # Density map decoder regresssion module specifics.

        self.decode_head0 = nn.Sequential(
            nn.Conv2d(fim_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

        # self.shot_token = nn.Parameter(torch.zeros(768))
        # self.blocks = nn.ModuleList([
        #     Block(fim_embed_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(fim_depth)])
        # self.norm = norm_layer(fim_embed_dim)

        # self.decoder_blocks = nn.ModuleList([
        #     CrossAttentionBlock(fim_embed_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(fim_embed_dim)])

        # --------------------------------------------------------------------------

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # CLIP model specifics (contains image and text encoder modules).

        self.clip_model = open_clip.create_model(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )

        self.dinov2_vitb14 = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14'
        ) 

        # Freeze all the weights of the text encoder.
        vis_copy = copy.deepcopy(self.clip_model.visual)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.visual = vis_copy

    def initialize_weights(self):
        # Initialize the positional embedding for the feature interaction module.
        fim_pos_embed = get_2d_sincos_pos_embed(
            self.fim_pos_embed.shape[-1],
            int(self.fim_num_img_tokens**0.5),
            cls_token=False,
        )
        self.fim_pos_embed.data.copy_(
            torch.from_numpy(fim_pos_embed).float().unsqueeze(0)
        )

        # Initialize nn.Linear and nn.LayerNorm layers.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use Xavier uniform weight initialization following the official JAX ViT.
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_img_encoder(self, imgs):
        return self.clip_model.encode_image(imgs)
    
    def forward_dino_img_encoder(self, imgs):
        # Define the image preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
                # transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(degrees=30),
                # transforms.RandomBrightness(),
                # transforms.RandomContrast(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocess the images
        imgs = [preprocess(img) for img in imgs]

        batch = torch.stack(imgs)
        batch = batch.cuda()

        out = self.dinov2_vitb14(batch, is_training=True)        
        # print(out.keys()) # dict_keys(['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks'])

        out = out["x_norm_patchtokens"]

        return out


    def foward_txt_encoder(self, counting_queries):
        return self.clip_model.encode_text(counting_queries)

    # def forward_fim(self, img_tokens, txt_tokens):
    #     # Add positional embedding to image tokens.
    #     img_tokens = img_tokens + self.fim_pos_embed

    #     # Pass image tokens and counting query tokens through the feature interaction module.
    #     x = img_tokens
    #     for blk in self.fim_blocks:
    #         x = blk(x, txt_tokens)

    #     return self.fim_norm(x)
    
    def forward_fim(self, img_tokens):
        # Add positional embedding to image tokens.
        x = img_tokens + self.fim_pos_embed

        # txt_tokens = torch.zeros_like(x)

        # for blk in self.fim_blocks:
        #     x = blk(x, txt_tokens)

        # # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        return self.fim_norm(x)

    def forward_decoder(self, fim_output_tokens):
        # Reshape the tokens output by the feature interaction module into a square feature map with [fim_embed_dim] channels.
        n, hw, c = fim_output_tokens.shape
        h = w = int(math.sqrt(hw))
        x = fim_output_tokens.transpose(1, 2).reshape(n, c, h, w)

        # y = self.shot_token.unsqueeze(0).to(x.device)
        # y = y.transpose(0,1) # y [3,N,C]->[N,3,C]

        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x, y)
        # x = self.decoder_norm(x)

        # Upsample output of this map to be N x [fim_embed_dim] x 24 x 24, as it was in CounTR.
        x = F.interpolate(x, size=24, mode="bilinear", align_corners=False)

        # Pass [x] through the density map regression decoder and upsample output until density map is the size of the input image.
        x = F.interpolate(
            self.decode_head0(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head1(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head2(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head3(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )

        # Remove the channel dimension from [x], as the density map only has 1 channel.
        return x.squeeze(-3)

    def forward(self, imgs, counting_queries):
        img_tokens = self.forward_dino_img_encoder(imgs)
        # Add a token dimension to the CLIP text embeddings.
        txt_tokens = self.foward_txt_encoder(counting_queries).unsqueeze(-2)
        # fim_output_tokens = self.forward_fim(img_tokens, txt_tokens)
        fim_output_tokens = self.forward_fim(img_tokens)
        pred = self.forward_decoder(fim_output_tokens)
        return pred
