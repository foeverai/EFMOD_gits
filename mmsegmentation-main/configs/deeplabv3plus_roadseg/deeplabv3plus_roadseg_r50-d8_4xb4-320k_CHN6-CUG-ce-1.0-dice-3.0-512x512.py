_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_road.py', '../_base_/datasets/chn6cug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_adamw_320k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2,
                     loss_decode=[
                         dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
                     ]
                     ),
    auxiliary_head=dict(num_classes=2))
