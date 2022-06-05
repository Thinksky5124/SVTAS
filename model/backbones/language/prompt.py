'''
Author       : Thyssen Wen
Date         : 2022-06-03 10:42:44
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-04 21:30:09
Description  : Prompt Module ref:https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
FilePath     : /ETESVS/model/backbones/language/prompt.py
'''
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from torch.nn import functional as F
from utils.logger import get_logger
from torch.nn.utils.rnn import pad_sequence

from ...builder import BACKBONES
# from clip import clip
from ..utils.clip import LayerNorm
from ..utils.clip import SimpleTokenizer as _Tokenizer
from ..utils.clip import Transformer
from ..utils.transducer import get_attn_pad_mask


@BACKBONES.register()
class LearnerTextEncoder(nn.Module):
    def __init__(self,
                 actions_map_file_path,
                 vocab_size,
                 embedding_dim,
                 encoder_layers_num,
                 encoder_heads_num,
                 text_embed_dim,
                 max_seg_action_num=12,
                 max_len=256,
                 n_ctx=8,
                 ctx_init="",
                 class_token_position="end",
                 ignore_index=-100,
                 pretrained=None,
                 context_init_method="class_specific"):
        super().__init__()
        attn_mask = None
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        classnames = [a.split()[1] for a in actions]
        id2classes = {int(a.split()[0]): a.split()[1] for a in actions}

        self.pretrained = pretrained

        self.prompt_learner = PromptLearner(classnames, vocab_size, embedding_dim, max_seg_action_num=max_seg_action_num,
            n_ctx=n_ctx, ctx_init=ctx_init, class_token_position=class_token_position, labels_id2classesname=id2classes,
            ignore_index=ignore_index, context_init_method=context_init_method, max_len=max_len)
        self.transformer = Transformer(width=embedding_dim, layers=encoder_layers_num, heads=encoder_heads_num, attn_mask=attn_mask)
        self.positional_embedding = nn.Parameter(torch.empty(max_len, embedding_dim))
        self.ln_final = LayerNorm(embedding_dim)
        self.text_projection = nn.Linear(embedding_dim, text_embed_dim)

    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)
            else:
                nn.init.normal_(self.positional_embedding, std=0.01)
                nn.init.normal_(self.prompt_learner.token_embedding.weight, std=0.02)
        else:
            nn.init.normal_(self.positional_embedding, std=0.01)
            nn.init.normal_(self.prompt_learner.token_embedding.weight, std=0.02)

    def forward(self, last_clip_labels, masks):
        b, _ = masks.shape
        prompts, pad_masks = self.prompt_learner(last_clip_labels, b)
        prompts = prompts.to(masks.device)
        pad_masks = pad_masks.to(masks.device)

        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        x = self.text_projection(x)

        return x


class PromptLearner(nn.Module):
    def __init__(self,
                 classnames,
                 vocab_size,
                 embedding_dim,
                 max_seg_action_num=12,
                 max_len=256,
                 n_ctx=8,
                 ctx_init="",
                 class_token_position="end",
                 labels_id2classesname=None,
                 ignore_index=-100,
                 context_init_method="class_specific"):
        super().__init__()
        self._tokenizer = _Tokenizer(50)
        self.max_seg_action_num = max_seg_action_num
        self.max_len = max_len
        n_cls = len(classnames)
        self.id2classes = labels_id2classesname
        self.ignore_index = ignore_index
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        logger = get_logger("ETESVS")

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
            if context_init_method == "class_specific":
                logger.info("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, embedding_dim)
            else:
                logger.info("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, embedding_dim)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        self.order_prefix = [f"Firstly, We have watch {{}} frames, ", f"Secondly, We have watch {{}} frames, ",
                             f"Thirdly, We have watch {{}} frames, ", f"Fourthly, We have watch {{}} frames, ",
                             f"Fifthly, We have watch {{}} frames, ", f"Sixthly, We have watch {{}} frames, ",
                             f"Seventhly, We have watch {{}} frames, ", f"Eighthly, We have watch {{}} frames, ",
                             f"Ninthly, We have watch {{}} frames, ", f"Tenthly, We have watch {{}} frames, ",
                             f"Eleventhly, We have watch {{}} frames, ", f"Twelfthly, We have watch {{}} frames, "]
        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # (n_cls, n_tkn)
        tokenized_prompt_prefix = self._tokenizer.tokenize(prompt_prefix)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompt_prefix)

        # SOS
        self.token_prefix = embedding[:, :1, :]
        # EOS
        self.token_suffix = embedding[:, -1:, :]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.max_len = max_len
        self.class_token_position = class_token_position
    
    def convert_id_to_promot(self, label_idx_tensor):
        # label_idx_tensor [T]
        # promot [ctx_len D]

        labels_idx_order, counts = torch.unique_consecutive(label_idx_tensor, return_counts=True)
        labels_idx_order = list(labels_idx_order.detach().cpu().numpy())
        counts = list(counts.detach().cpu().numpy())
        promot_embedding = [self.token_prefix.to(label_idx_tensor.device)]
        for idx in range(self.max_seg_action_num):
            if idx < len(labels_idx_order):
                label_idx = labels_idx_order[idx]
                label_name = self.id2classes[label_idx]
                
                promot_prefix = self._tokenizer.tokenize(self.order_prefix[idx].format(str(counts[idx])))[:, 1:11].to(label_idx_tensor.device)
                # [N 10 D]
                promot_prefix_embedding = self.token_embedding(promot_prefix)
                # [N 8 D]
                learner_promot = self.ctx
                if learner_promot.dim() == 2:
                    learner_promot = learner_promot.unsqueeze(0).expand(self.n_cls, -1, -1)

                token_labels_len = len(self._tokenizer.encode(label_name))
                # [N (token_labels_len + 1) D]
                label_promot = self._tokenizer.tokenize(label_name + ".")[:, 1:(2 + token_labels_len)].to(label_idx_tensor.device)
                label_promot_embedding = self.token_embedding(label_promot)
            else:
                label_name = "No action"
                promot_prefix = self._tokenizer.tokenize(self.order_prefix[idx].format(str(0)))[:, 1:11].to(label_idx_tensor.device)
                # [N 10 D]
                promot_prefix_embedding = self.token_embedding(promot_prefix)
                # [N 8 D]
                learner_promot = self.ctx
                if learner_promot.dim() == 2:
                    learner_promot = learner_promot.unsqueeze(0).expand(self.n_cls, -1, -1)

                token_labels_len = len(self._tokenizer.encode(label_name))
                # [N (token_labels_len + 1) D]
                label_promot = self._tokenizer.tokenize(label_name + ".")[:, 1:(2 + token_labels_len)].to(label_idx_tensor.device)
                label_promot_embedding = self.token_embedding(label_promot)

            if self.class_token_position == "end":
                token_embedding = torch.cat([
                    promot_prefix_embedding,
                    learner_promot[label_idx:(label_idx + 1)],
                    label_promot_embedding], dim=1)
            elif self.class_token_position == "middle":
                half_n_ctx = learner_promot // 2
                dot_vector = label_promot_embedding[:, -1:]
                ctx_i_half1 = learner_promot[label_idx:(label_idx + 1), :half_n_ctx, :]
                ctx_i_half2 = learner_promot[label_idx:(label_idx + 1), half_n_ctx:, :]
                token_embedding = torch.cat([
                    promot_prefix_embedding,
                    ctx_i_half1,
                    label_promot_embedding[:, :-1],
                    ctx_i_half2,
                    dot_vector], dim=1)
            elif self.class_token_position == "front":
                dot_vector = label_promot_embedding[:, -1:]
                token_embedding = torch.cat([
                    label_promot_embedding[:, :-1],
                    learner_promot[label_idx:(label_idx + 1)],
                    promot_prefix_embedding,
                    dot_vector], dim=1)
            else:
                raise ValueError
            promot_embedding.append(token_embedding)
        
        promot_embedding = torch.cat(promot_embedding, dim=1)
        return promot_embedding

    def forward(self, last_clip_labels, batch_size):
        if last_clip_labels is None:
            prompts = self.token_prefix.expand(batch_size, self.max_len, -1)
        else:
            text_list = []
            for b in range(batch_size):
                if torch.any(last_clip_labels[b,:] == self.ignore_index):
                    embedding = self.token_suffix.expand(-1, self.max_len, -1)
                else:
                    embedding = self.convert_id_to_promot(last_clip_labels[b, :])
                text_list.append(embedding.squeeze(0))
            
            # [N U D]
            prompts = pad_sequence(text_list, batch_first=True)
        if prompts.shape[-2] < self.max_len:
            prompts = F.pad(prompts, pad=[0, 0, 0, self.max_len - prompts.shape[-2], 0, 0], mode='constant', value=0.0)
        else:
            prompts = prompts[:, :self.max_len, :]
        pad_masks = torch.where(prompts != 0., torch.ones_like(prompts), torch.zeros_like(prompts))
        return prompts, pad_masks
