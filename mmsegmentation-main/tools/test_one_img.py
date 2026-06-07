# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('save_dir', default=None, help='save result dir')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # 获取文件名
    img_name = Path(args.img_path).name

    result = inference_model(model,args.img_path)

    # 提取预测的语义分割掩码数据 (H, W)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # 将像素值为1（前景）的地方改为255（纯白），0（背景）保持为0（纯黑）
    binary_mask = (pred_mask * 255).astype(np.uint8)

    # 保存纯黑白图片
    outbw_file = os.path.join(args.save_dir, img_name)
    cv2.imwrite(outbw_file, binary_mask)


    out_file = os.path.join(args.out_file,img_name)
    # show the results
    show_result_pyplot(
        model,
        args.img_path,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=True,
        show=False if args.out_file is not None else True,
        out_file=out_file,
        save_dir=args.save_dir
    )



if __name__ == '__main__':
    main()
