'''
Author       : Thyssen Wen
Date         : 2022-10-26 10:15:16
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 15:33:33
Description  : ActionCLIP TextCLIP ref:https://github.com/sallymmx/ActionCLIP/blob/master/modules/Text_Prompt.py
FilePath     : /SVTAS/svtas/model/backbones/language/text_clip.py
'''
# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import torch.nn as nn
import numpy
from svtas.utils import AbstractBuildFactory
from ..utils.clip import SimpleTokenizer as _Tokenizer

@AbstractBuildFactory.register('model')
class TextCLIP(nn.Module):
    def __init__(self,
                 clip_model,
                 actions_map_file_path,
                 max_len=77) -> None:
        super().__init__()
        self.clip_model = clip_model
        self._tokenizer = _Tokenizer(max_len)
        self.text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
        self.num_text_aug = len(self.text_aug)
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        id2classes = {int(a.split()[0]): a.split()[1] for a in actions}
        self.text_dict = {}
        for ii, txt in enumerate(self.text_aug):
            self.text_dict[ii] = torch.cat([self._tokenizer.tokenize(txt.format(c)) for i, c in id2classes.items()])
    
    def _clear_memory_buffer(self):
        pass

    def init_weights(self, child_model=False, revise_keys=[(r'backbone.', r'')]):
        pass

    def __call__(self, labels, masks):
        text_id = numpy.random.randint(self.num_text_aug, size = labels.shape[0])
        texts = torch.stack([self.text_dict[j][i,:] for i,j in zip(labels, text_id)])
        text_embedding = self.clip_model.encode_text(texts)
        return text_embedding
    