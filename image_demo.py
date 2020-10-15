import tensorflow as tf
import cv2
import time
import argparse
import os
from posenet.posenet_factory import load_model

from tfjs_graph_converter import api

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')  # mobilenet resnet50
parser.add_argument('--stride', type=int, default=16)  # 8, 16, 32 (max 16 for mobilenet)
parser.add_argument('--quant_bytes', type=int, default=4)  # 4 = float
parser.add_argument('--multiplier', type=float, default=1.0)  # only for mobilenet
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():

    print('Tensorflow version: %s' % tf.__version__)
    assert tf.__version__.startswith('2.'), "Tensorflow version 2.x must be used!"

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    model = args.model  # mobilenet resnet50
    stride = args.stride  # 8, 16, 32 (max 16 for mobilenet, min 16 for resnet50)
    quant_bytes = args.quant_bytes  # float
    multiplier = args.multiplier  # only for mobilenet

    posenet = load_model(model, stride, quant_bytes, multiplier)

    filenames = [f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    for f in filenames:
        img = cv2.imread(f)
        pose_scores, keypoint_scores, keypoint_coords = posenet.estimate_multiple_poses(img)
        img_poses = posenet.draw_poses(img, pose_scores, keypoint_scores, keypoint_coords)
        posenet.print_scores(f, pose_scores, keypoint_scores, keypoint_coords)
        cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), img_poses)

    print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
