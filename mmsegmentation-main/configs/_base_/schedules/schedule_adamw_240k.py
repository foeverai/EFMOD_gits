# optimizer
optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05, eps=1e-08, betas=(0.9,0.999,))
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
optim_wrapper=dict(type='OptimWrapper',optimizer=optimizer,clip_grad=None,paramwise_cfg=dict(norm_decay_mult=0.0))
# learning policy
param_scheduler=[
dict(type='LinearLR',start_factor=0.001,by_epoch=False,begin=0,end=500),
dict(type='PolyLR',eta_min=0.0002,power=0.9,begin=500,end=240000,by_epoch=False)]
# training schedule for 240k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=240000, val_interval=12000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=12000,max_keep_ckpts=5,save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
