'''
Author       : Thyssen Wen
Date         : 2022-10-28 10:49:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 10:52:50
Description  : ActionCLIP
FilePath     : /SVTAS/config/_base_/models/action_recognition/action_clip.py
'''
MODEL = dict(
    architecture = "ActionCLIP",
    pretrained = "./data/vit-16-32f.pt",
    is_feature_extract = True,
    image_prompt = dict(
        name = "CLIP",
        # pretrained = "./data/ViT-B-16.pt",
        embed_dim = 512,
        image_resolution = 224,
        vision_layers = 12,
        vision_width = 768,
        vision_patch_size = 16,
        context_length = 77,
        vocab_size = 49408,
        transformer_width = 512,
        transformer_heads = 8,
        transformer_layers = 12,
        joint=False,
        tsm=False,
        clip_seg_num=8,
        dropout = 0.,
        emb_dropout = 0.,
    ),
    text_prompt = dict(
        name = "TextCLIP"
    ),
    fusion_neck = dict(
        name = "ActionCLIPFusionNeck",
        sim_head = "Transf",
        embed_dim_cfg = 512,
        context_length_cfg = 77,
        transformer_width_cfg = 512,
        clip_seg_num = 8
    ),
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 512,
        input_seg_num = 8,
        output_seg_num = 1,
        sample_rate = 8,
        pool_space = True,
        in_format = "N*T,C",
        out_format = "NCT"
    ),
    loss = None
)