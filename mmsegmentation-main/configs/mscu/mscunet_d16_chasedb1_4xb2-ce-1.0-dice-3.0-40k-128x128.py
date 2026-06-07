_base_ = [
    '../_base_/models/mscu_net.py',
    '../_base_/datasets/chase_db1_pad.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (128, 128)
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