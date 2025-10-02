import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import cv2
import math

sys.path.append('.')
import config as cfg
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help='Input directory')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--n', type=int, default=4, help='Nx')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    n = args.n

    os.makedirs(out_dir, exist_ok=True)

    TTA = True
    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 3, 3]
    )
    down_scale = 1.0

    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()

    def _recursive_generator(frame1, frame2, down_scale, num_recursions, index):
        if num_recursions == 0:
            yield frame1, index
        else:
            mid_frame = model.inference(frame1, frame2, True, TTA=TTA, fast_TTA=TTA, scale=0.0)
            id = 2 ** (num_recursions - 1)
            yield from _recursive_generator(frame1, mid_frame, down_scale, num_recursions - 1, index - id)
            yield from _recursive_generator(mid_frame, frame2, down_scale, num_recursions - 1, index + id)

    img_name_list = os.listdir(in_dir)
    img_name_list.sort()

    n_imgs = len(img_name_list)
    n_iters = n_imgs - 1

    cur_idx = 0
    for i in tqdm(range(n_iters)):
        img_0 = cv2.imread(os.path.join(in_dir, img_name_list[i]))
        img_1 = cv2.imread(os.path.join(in_dir, img_name_list[i + 1]))

        img_0 = (torch.tensor(img_0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
        img_1 = (torch.tensor(img_1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

        padder = InputPadder(img_0.shape, divisor=32)
        img_0, img_1 = padder.pad(img_0, img_1)

        frames = list(_recursive_generator(img_0, img_1, down_scale, int(math.log2(n)), n // 2))
        frames = sorted(frames, key=lambda x: x[1])
        ans = []

        for pred, _ in frames:
            pred = pred[0]
            pred = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            ans.append(pred)

        for img in ans:
            cv2.imwrite(os.path.join(out_dir, f"{cur_idx:05d}.png"), img)
            cur_idx += 1


if __name__ == '__main__':
    main()