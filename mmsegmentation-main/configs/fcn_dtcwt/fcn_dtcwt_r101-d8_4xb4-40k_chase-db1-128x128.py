_base_ = './fcn_dtcwt_r50-d8_4xb4-40k_chase-db1-128x128.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
