_base_ = [
    '../_base_/models/efemod.py',
    '../_base_/datasets/map.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))