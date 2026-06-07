_base_ = './upernet_r50_4xb4-40k_chase-db1-128x128.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))




