_base_ = [
    '../_base_/models/afdanet.py', '../_base_/datasets/deep_globe_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_adamw_320k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=crop_size, stride=(340, 340),),
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
    )
)
