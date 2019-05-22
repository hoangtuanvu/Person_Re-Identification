import os
import argparse
from re_id.reid.utils.data.iotools import mkdir_if_missing


def parse_images_to_video(img_dir, args):
    dir_name = img_dir.split('/')[-1]
    os.system("ffmpeg -framerate {} -pattern_type glob -i '{}/*.{}' -c:v {} {}/{}.{}"
              .format(args.frame_rate,  # FPS
                      img_dir,  # Folder which contains images,
                      args.img_ext,  # Image Extension,
                      args.codec,  # Codec for compression
                      args.output_dir,  # Folder which contains videos after compression
                      dir_name,  # Video name
                      args.vid_ext))  # Video Extension


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', dest='images_dir',
                        type=str, help='Directory of images which transformed to videos')
    parser.add_argument('--output-dir', default='videos',
                        type=str, help='Directory of output videos')
    parser.add_argument('--f', dest='frame_rate', default=10,
                        type=int, help='FPS or Frames Per Second')
    parser.add_argument('--c', dest='codec', default='libx264',
                        type=str, help='codec type for saving videos')
    parser.add_argument('--img-ext', default='jpg',
                        type=str, help='image extension')
    parser.add_argument('--vid-ext', default='mp4',
                        type=str, help='video extension')

    args = parser.parse_args()

    # create output directory if it does not exist
    mkdir_if_missing(args.output_dir)

    for sub_dir in os.listdir(args.images_dir):
        parse_images_to_video(os.path.join(args.images_dir, sub_dir), args)
