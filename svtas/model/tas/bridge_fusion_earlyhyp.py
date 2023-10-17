'''
Author       : Thyssen Wen
Date         : 2022-06-15 16:13:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 16:18:21
Description  : Bridge-Prompt Fusion Model ref:https://github.com/ttlmh/Bridge-Prompt/blob/master/modules/fusion_module.py
FilePath     : /SVTAS/svtas/model/necks/bridge_fusion_earlyhyp.py
'''
import torch
from torch import nn
from svtas.utils import AbstractBuildFactory

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def trunc_normal_(x, mean=0., std=1.):
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class BPromptFusing(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6, heads=8):
        super(BPromptFusing, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cnt_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 3, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cnt_token, std=.02)
            trunc_normal_(self.sep_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ord_emb):
        b, n, f, c = x.shape
        x = x.view(-1, f, c)
        ord_emb = ord_emb.view(-1, ord_emb.shape[-1])
        ord_emb = torch.unsqueeze(ord_emb, dim=1)
        nvids = x.shape[0]

        cnt_token = self.cnt_token.expand(nvids, -1, -1)
        sep_token = self.sep_token.expand(nvids, -1, -1)
        x = torch.cat((cnt_token, ord_emb, sep_token, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        x = self.transformer_enc(x)
        x.transpose_(1, 0)
        x = x.view(b, n, -1, c)
        return x[:, :, 0], x[:, :, -self.clip_length:]


@AbstractBuildFactory.register('model')
class BridgePromptFusionEarlyhyp(nn.Module):
    def __init__(self,
                 embedding_dim=512,
                 num_layers=6,
                 cnt_max=7,
                 context_length=77,
                 transformer_width=512,
                 clip_seg_num=32):
        super().__init__()
        self.cnt_max = cnt_max
        self.clip_seg_num = clip_seg_num

        transformer_width = transformer_width
        transformer_heads = transformer_width // 64

        self.frame_position_embeddings = nn.Embedding(context_length, embedding_dim)

        self.pos_emb = nn.Sequential(nn.LayerNorm(embedding_dim),
                                        nn.Linear(embedding_dim, clip_seg_num),
                                        nn.Softmax(dim=-1))

        self.transformer = BPromptFusing(clip_length=self.clip_seg_num, embed_dim=embedding_dim, n_layers=num_layers,
                                            heads=transformer_heads)

    def init_weights(self, init_cfg: dict = {}):
        self.apply(self._init_weights)

    def _clear_memory_buffer(self):
        pass

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, img_inputs, text_inputs, masks):
        ### img_inputs img_feature
        # img_feature [N C T]
        # masks [N T]
        ### text_inputs [text_all_embedding, text_cnt_embedding, text_acts_embedding, text_pos_embedding]
        # text_all_embedding [B D]
        # text_cnt_embedding [B D]
        # text_acts_embedding [B cnt_max D]
        # text_pos_embedding [B pos_cnt D]
        img_seg_feature = img_inputs
        text_all_embedding, text_cnt_embedding, text_acts_embedding, text_pos_embedding = text_inputs
        
        # [N C T] -> [N cnt_max C T] -> [N cnt_max T C]
        img_feature = img_seg_feature.unsqueeze(1).repeat(1, self.cnt_max, 1, 1)
        img_feature = torch.permute(img_feature, dims=[0, 1, 3, 2])
        b, n, t, c = img_feature.size()
        x = img_feature.contiguous()

        x_original = x
        cnt_emb, x = self.transformer(x, text_pos_embedding)
        cnt_emb = cnt_emb.type(x_original.dtype)
        x = x.type(x_original.dtype) + x_original

        seg_feature = x.mean(dim=2, keepdim=False)

        text_feature = text_all_embedding, text_cnt_embedding, text_acts_embedding, \
                    cnt_emb.mean(dim=1, keepdim=False)
        
        return text_feature, seg_feature
