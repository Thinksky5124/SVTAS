'''
Author       : Thyssen Wen
Date         : 2022-10-26 10:15:16
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-26 20:34:10
Description  : ActionCLIP TextCLIP ref:https://github.com/sallymmx/ActionCLIP/blob/master/modules/Text_Prompt.py
FilePath     : /SVTAS/model/backbones/language/text_clip.py
'''
# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
from ...builder import BACKBONES
from ..utils.clip import SimpleTokenizer as _Tokenizer

@BACKBONES.register()
class TextCLIP(object):
    def __init__(self,
                 max_len=77) -> None:
        self._tokenizer = _Tokenizer(max_len)
        self.text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
        self.num_text_aug = len(self.text_aug)
    
    def _clear_memory_buffer(self):
        pass

    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        pass

    def __call__(self, labels, masks):
        text_dict = {}
        num_text_aug = len(self.text_aug)

        for ii, txt in enumerate(self.text_aug):
            text_dict[ii] = torch.cat([self._tokenizer(txt.format(c)) for i, c in data.classes])

        text_id = numpy.random.randint(self.num_text_aug, size = labels.shape[0])
        texts = torch.stack([text_dict[j][i,:] for i,j in zip(labels.shape[0], text_id)])

        return texts
    