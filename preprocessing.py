import os.path as osp
import cv2
import os
import argparse
import glob
from re_id.reid.utils.data.iotools import mkdir_if_missing


def parse_images_to_video(dir, out_dir):
    if not osp.exists(dir):
        raise ValueError('Images folder does not exist!')

    dir_name = dir.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(out_dir, '{}.mp4'.format(dir_name)), fourcc, 10.0, (1920, 1080))

    imgs_dir = glob.glob('{}/*.jpg'.format(dir))
    imgs_dir.sort()

    for img_path in imgs_dir:
        out.write(cv2.imread(img_path))

    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', dest='images_dir',
                        type=str, help='Directory of images which transformed to videos')
    parser.add_argument('--output-dir', default='videos',
                        type=str, help='Directory of output videos')

    args = parser.parse_args()

    # create output directory if it does not exist
    mkdir_if_missing(args.output_dir)

    for sub_dir in os.listdir(args.images_dir):
        parse_images_to_video(os.path.join(args.images_dir, sub_dir), args.output_dir)
