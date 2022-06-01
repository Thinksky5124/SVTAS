'''
Author       : Thyssen Wen
Date         : 2022-05-21 19:53:44
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-27 15:38:06
Description  : COCOOP Prompt Module ref:https://github.com/KaiyangZhou/CoOp/blob/main/trainers/cocoop.py
FilePath     : /ETESVS/model/backbones/language/cocoop_prompt.py
'''
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from utils.logger import get_logger
from mmcv.runner import load_checkpoint
from ...builder import BACKBONES


# from clip import clip
from ..utils.clip import SimpleTokenizer as _Tokenizer
from ..utils.clip import Transformer, LayerNorm
from ..utils.transducer import get_attn_pad_mask

@BACKBONES.register()
class COCOOPTextEncoder(nn.Module):
    def __init__(self,
                 actions_map_file_path,
                 vocab_size,
                 transformer_width,
                 encoder_layers_num,
                 encoder_heads_num,
                 text_embed_dim,
                 vis_dim=2048,
                 ctx_dim=77,
                 n_ctx=8,
                 ctx_init=""):
        super().__init__()
        attn_mask = None
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        classnames = [a.split()[1] for a in actions]
        
        self.prompt_learner = PromptLearner(classnames, vocab_size, transformer_width, vis_dim=vis_dim, ctx_dim=ctx_dim, n_ctx=n_ctx, ctx_init=ctx_init)
        self.transformer = Transformer(width=transformer_width, layers=encoder_layers_num, heads=encoder_heads_num, attn_mask=attn_mask)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, text_embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)

    def forward(self, img_features, label):
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        prompts = self.prompt_learner(img_features)
        logit_scale = self.logit_scale.exp()

        logits = []
        for pts_i, imf_i in zip(prompts, img_features):
            x = pts_i + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

            text_features = x / x.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)
        
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return l_i


class PromptLearner(nn.Module):
    def __init__(self, classnames, vocab_size, transformer_width, vis_dim=2048, ctx_dim=77, n_ctx=8, ctx_init=""):
        super().__init__()
        self._tokenizer = _Tokenizer(ctx_dim)
        n_cls = len(classnames)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = self._tokenizer.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self._tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([self._tokenizer.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts