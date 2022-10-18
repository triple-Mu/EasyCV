CLASSES = ['Bacillariophyta', 'Chlorella', 'Chrysophyta',
           'Dunaliella_salina', 'Platymonas', 'translating_Symbiodinium',
           'bleaching_Symbiodinium', 'normal_Symbiodinium']


img_scale = [(1280, 1920), (1344, 1920), (1408, 1920), (1472, 1920),
             (1536, 1920), (1600, 1920), (1664, 1920), (1728, 1920),
             (1792, 1920), (1856, 1920)]

# dataset settings
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(
        type='MMAutoAugment',
        policies=[
            [
                dict(
                    type='MMResize',
                    img_scale=img_scale,
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='MMResize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(1280, 2880), (1600, 2880), (1920, 2880)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='MMRandomCrop',
                    crop_type='absolute_range',
                    crop_size=(1536, 1728),
                    allow_negative_crop=True),
                dict(
                    type='MMResize',
                    img_scale=img_scale,
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'ori_img_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'img_norm_cfg'))
]
test_pipeline = [
    dict(
        type='MMMultiScaleFlipAug',
        img_scale=(1600, 1600),
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='MMPad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'ori_img_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'))
        ])
]

train_dataset = dict(
    type='DetDataset',
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        test_mode=False,
        filter_empty_gt=False,
        iscrowd=False),
    pipeline=train_pipeline)

val_dataset = dict(
    type='DetDataset',
    imgs_per_gpu=1,
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        test_mode=True,
        filter_empty_gt=False,
        iscrowd=True),
    pipeline=test_pipeline)

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=train_dataset,
    val=val_dataset,
    drop_last=True)

# evaluation
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        dist_eval=True,
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
        ],
    )
]
