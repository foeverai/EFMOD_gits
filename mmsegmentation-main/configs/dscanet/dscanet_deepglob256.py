_base_ = [
    '../_base_/models/dsca.py', '../_base_/datasets/deep_globe_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_adamw_640k.py'
]
crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(224, 224), stride=(150,150 )))
