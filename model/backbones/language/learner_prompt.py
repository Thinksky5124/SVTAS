'''
Author       : Thyssen Wen
Date         : 2022-06-03 10:42:44
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-16 09:59:36
Description  : Prompt Module ref:https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
FilePath     : /ETESVS/model/backbones/language/learner_prompt.py
'''

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from torch.nn import functional as F
from utils.logger import get_logger
from torch.nn.utils.rnn import pad_sequence
from num2words import num2words

from ...builder import BACKBONES
# from clip import clip
from ..utils.clip import LayerNorm
from ..utils.clip import SimpleTokenizer as _Tokenizer
from ..utils.clip import Transformer
from ..utils.transducer import get_attn_pad_mask


@BACKBONES.register()
class LearnerPromptTextEncoder(nn.Module):
    def __init__(self,
                 actions_map_file_path,
                 embedding_dim,
                 encoder_layers_num,
                 encoder_heads_num,
                 text_embed_dim,
                 vocab_size=49408,
                 sample_rate=4,
                 clip_seg_num=32,
                 max_len=40,
                 n_ctx=8,
                 ctx_init="",
                 class_token_position="end",
                 ignore_index=-100,
                 pretrained=None,
                 token_embedding_pretrained=None,
                 context_init_method="class_specific"):
        super().__init__()
        attn_mask = None
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        classnames = [a.split()[1] for a in actions]
        id2classes = {int(a.split()[0]): a.split()[1] for a in actions}

        self.pretrained = pretrained
        self.token_embedding_pretrained = token_embedding_pretrained
        self.sample_rate = sample_rate

        self.prompt_learner = PromptLearner(classnames=classnames, embedding_dim=embedding_dim, vocab_size=vocab_size, n_ctx=n_ctx,
            ctx_init=ctx_init, class_token_position=class_token_position, labels_id2classesname=id2classes, ignore_index=ignore_index,
            context_init_method=context_init_method, max_len=max_len, sample_rate=sample_rate)
        self.transformer = Transformer(width=embedding_dim, layers=encoder_layers_num, heads=encoder_heads_num, attn_mask=attn_mask)
        self.positional_embedding = nn.Parameter(torch.empty(clip_seg_num, max_len, embedding_dim))
        self.ln_final = LayerNorm(embedding_dim)
        self.text_projection = nn.Linear(embedding_dim, text_embed_dim)
        self.squeeze_sentence = nn.Linear(max_len * text_embed_dim, text_embed_dim)

    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("SVTAS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)
            else:
                nn.init.normal_(self.positional_embedding, std=0.01)
                nn.init.normal_(self.prompt_learner.token_embedding.weight, std=0.02)
        else:
            nn.init.normal_(self.positional_embedding, std=0.01)
            nn.init.normal_(self.prompt_learner.token_embedding.weight, std=0.02)
        
        if isinstance(self.token_embedding_pretrained, str):
            logger = get_logger("SVTAS")
            load_checkpoint(self, self.token_embedding_pretrained, strict=False, logger=logger, revise_keys=[(r'token_embedding', r'prompt_learner.token_embedding')])

    def forward(self, labels, masks):
        b, temporal_len = masks.shape
        prompts, pad_masks = self.prompt_learner(labels, b, temporal_len, masks.device)
        # [N T U D]
        prompts = prompts.to(masks.device)
        # [N T U 1] -> [N*T U 1]
        pad_masks = pad_masks.reshape([-1] + list(pad_masks.shape[2:])).to(masks.device)

        x = prompts + self.positional_embedding
        # [N T U D] -> [N*T U D]
        x = torch.reshape(x, [-1] + list(prompts.shape[2:]))
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        x = self.text_projection(x)
        # [N*T U D] -> [N T U D]
        x = torch.reshape(x, [-1, temporal_len // self.sample_rate] + list(x.shape[1:]))
        # [N T U D] -> [N T U*D] -> [N T D] -> [N D T]
        x = torch.reshape(x, list(x.shape[:2]) + [-1])
        x = self.squeeze_sentence(x)
        x = torch.permute(x, [0, 2, 1])
        return x


class PromptLearner(nn.Module):
    def __init__(self,
                 classnames,
                 embedding_dim,
                 vocab_size=49408,
                 max_len=40,
                 sample_rate=4,
                 n_ctx=8,
                 ctx_init="",
                 class_token_position="end",
                 labels_id2classesname=None,
                 ignore_index=-100,
                 context_init_method="class_specific"):
        super().__init__()
        self._tokenizer = _Tokenizer(max_len)
        self.max_len = max_len
        self.sample_rate = sample_rate
        n_cls = len(classnames)
        self.id2classes = labels_id2classesname
        self.ignore_index = ignore_index
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        logger = get_logger("SVTAS")

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
                ctx_vectors = torch.normal(mean=0, std=0.2, size=(n_cls, n_ctx, embedding_dim))
            else:
                logger.info("Initializing a generic context")
                ctx_vectors = torch.normal(mean=0, std=0.2, size=(n_ctx, embedding_dim))
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        self.ordinal_prefix = {
            1 : "This is the first action, ", 2 : "This is the second action, ", 3 : "This is the third action, ", 4 : "This is the fourth action, ", 5 : "This is the fifth action, ",
            6 : "This is the sixth action, ", 7 : "This is the seventh action, ", 8 : "This is the eighth action, ", 9 : "This is the ninth action, ", 10 : "This is the tenth action, ",
            11 : "This is the eleventhly, ", 12 : "This is the twelfthly action, ", 13 : "This is the thirteenth action, ", 14 : "This is the fourteenth action, ", 15 : "This is the fifteenth action, ",
            16 : "This is the sixteenth, ", 17 : "This is the seventeenth action, ", 18 : "This is the eighteenth action, ", 19 : "This is the nineteenth action, ", 20 : "This is the twentieth action, ", 
            21 : "This is the twenty-first action, ", 22 : "This is the twenty-second action, ", 23 : "This is the twenty-third action, ", 24 : "This is the twenty-fourth action, ", 25 : "This is the twenty-fifth action, ", 
            26 : "This is the twenty-sixth action, ", 27 : "This is the twenty-seventh action, ", 28 : "This is the twenty-eighth action, ", 29 : "This is the twenty-ninth action, ", 30 : "This is the thirtieth action, ",
            31 : "This is the thirty-first action, ", 32 : "This is the thirty-second action, ", 33 : "This is the thirty-third action, ", 34 : "This is the thirty-fourth action, ", 35 : "This is the thirty-fifth action, ",
        }
        self.seg_len_prefix = f"This action lasted {{}} frames in current windows, "
        self.frames_position_prefix = f"This is frame {{}} of the action, "
        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.max_len = max_len
        self.class_token_position = class_token_position
    
    def convert_id_to_promot(self, label_idx_tensor):
        # label_idx_tensor [T // sample_rate]
        # promot [ctx_len D]

        # get seg_len_prefix number
        labels_idx_order, inverse_indices, counts = torch.unique_consecutive(label_idx_tensor, return_counts=True, return_inverse=True)
        labels_idx_order = list(labels_idx_order.detach().cpu().numpy())
        counts = list(counts.detach().cpu().numpy())

        promot_embedding = []
        for frame_idx in range(label_idx_tensor.shape[0]):
            order_idx = inverse_indices[frame_idx]
            label_idx = labels_idx_order[order_idx]
            label_name = self.id2classes[label_idx]

            promot_prefix_str = self.ordinal_prefix[int(order_idx) + 1] + self.seg_len_prefix.format(num2words(counts[int(order_idx)])) + \
                                self.frames_position_prefix.format(num2words(frame_idx + 1))
                     
            token_promot_prefix_len = len(self._tokenizer.encode(promot_prefix_str))
            promot_prefix = self._tokenizer.tokenize(promot_prefix_str).to(label_idx_tensor.device)
            promot_prefix_embedding_sos_eos = self.token_embedding(promot_prefix)
            # [N token_promot_prefix_len D]
            promot_prefix_embedding = promot_prefix_embedding_sos_eos[:, 1:(1 + token_promot_prefix_len)]
            token_prefix = promot_prefix_embedding_sos_eos[:, :1]
            token_suffix = promot_prefix_embedding_sos_eos[:, (1 + token_promot_prefix_len):(2 + token_promot_prefix_len)]
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
                    token_prefix,
                    promot_prefix_embedding,
                    learner_promot[label_idx:(label_idx + 1)],
                    label_promot_embedding,
                    token_suffix], dim=1)
            elif self.class_token_position == "middle":
                half_n_ctx = learner_promot // 2
                dot_vector = label_promot_embedding[:, -1:]
                ctx_i_half1 = learner_promot[label_idx:(label_idx + 1), :half_n_ctx, :]
                ctx_i_half2 = learner_promot[label_idx:(label_idx + 1), half_n_ctx:, :]
                token_embedding = torch.cat([
                    token_prefix,
                    promot_prefix_embedding,
                    ctx_i_half1,
                    label_promot_embedding[:, :-1],
                    ctx_i_half2,
                    dot_vector,
                    token_suffix], dim=1)
            elif self.class_token_position == "front":
                dot_vector = label_promot_embedding[:, -1:]
                token_embedding = torch.cat([
                    token_prefix,
                    label_promot_embedding[:, :-1],
                    learner_promot[label_idx:(label_idx + 1)],
                    promot_prefix_embedding,
                    dot_vector,
                    token_suffix], dim=1)
            else:
                raise ValueError
            if token_embedding.shape[-2] < self.max_len:
                token_embedding = F.pad(token_embedding, pad=[0, 0, 0, self.max_len - token_embedding.shape[-2], 0, 0], mode='constant', value=0.0)
            else:
                token_embedding = token_embedding[:, :self.max_len, :]
            promot_embedding.append(token_embedding)
        
        promot_embedding = torch.cat(promot_embedding, dim=0)
        return promot_embedding

    def forward(self, last_clip_labels, batch_size, temporal_len, device):
        if last_clip_labels is None:
            start_promot = self._tokenizer.tokenize("").to(device)
            start_promot_embedding = self.token_embedding(start_promot)
            prompts = start_promot_embedding[:, :1].expand(batch_size, temporal_len // self.sample_rate, self.max_len, -1)
        else:
            text_list = []
            for b in range(batch_size):
                if torch.any(last_clip_labels[b,:] == self.ignore_index):
                    end_promot = self._tokenizer.tokenize("").to(device)
                    end_promot_embedding = self.token_embedding(end_promot)
                    embedding = end_promot_embedding[:, 1:2].expand(1, temporal_len // self.sample_rate, self.max_len, -1)
                    text_list.append(embedding)
                else:
                    embedding = self.convert_id_to_promot(last_clip_labels[b, ::self.sample_rate])
                    text_list.append(embedding.unsqueeze(0))
            
            # [N T U D]
            prompts = torch.cat(text_list, dim=0)
        pad_masks = torch.where(prompts != 0., torch.ones_like(prompts), torch.zeros_like(prompts))[:, :, :, 0:1]
        return prompts, pad_masks
