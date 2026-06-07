_base_ = './psanet_r50-d8_4xb2-320k_deepglobe-512x512.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
