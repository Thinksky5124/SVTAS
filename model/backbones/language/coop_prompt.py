'''
Author       : Thyssen Wen
Date         : 2022-06-03 10:42:44
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-03 13:57:22
Description  : COOP Prompt Module ref:https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
FilePath     : /ETESVS/model/backbones/language/coop_prompt.py
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
class COOPTextEncoder(nn.Module):
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
                 ctx_init="",
                 class_token_position="end",
                 pretrained=None):
        super().__init__()
        attn_mask = None
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        classnames = [a.split()[1] for a in actions]

        self.pretrained = pretrained

        self.prompt_learner = PromptLearner(classnames, vocab_size, transformer_width, vis_dim=vis_dim, ctx_dim=ctx_dim,
            n_ctx=n_ctx, ctx_init=ctx_init, class_token_position=class_token_position)
        self.transformer = Transformer(width=transformer_width, layers=encoder_layers_num, heads=encoder_heads_num, attn_mask=attn_mask)
        self.positional_embedding = nn.Parameter(torch.empty(ctx_dim, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, text_embed_dim))

    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)

    def forward(self, img_features, label):
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        prompts = self.prompt_learner()

        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self,
                 classnames,
                 vocab_size,
                 transformer_width,
                 vis_dim=2048,
                 ctx_dim=77,
                 n_ctx=8,
                 ctx_init="",
                 class_token_position="end"):
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

        logger = get_logger("ETESVS")
        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self._tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # (n_cls, n_tkn)
        tokenized_prompts = torch.cat([self._tokenizer.tokenize(p) for p in prompts])
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
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts