import numpy as np
import cv2


def simple_overlay(bg, fg):
    bg[fg == 0] = 0
    return bg


def ave_overlay(bg, fg, weight=0.5):
    a = np.zeros_like(bg)
    a = bg * weight + fg * (1 - weight)
    a[fg == 0] = 0
    return a.astype(np.uint8)


def main(bg, fg, out_path):
    bg = cv2.imread(bg)
    fg = cv2.imread(fg)
    im = ave_overlay(bg, fg, weight=0.75)
    cv2.imwrite(out_path, im)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', type=str, help="background image")
    parser.add_argument('--fg', type=str, help="foreground image")
    parser.add_argument('--out_path', type=str, default="./out.png",
                        help="path to output image")
    args = parser.parse_args()

    bg = args.bg
    fg = args.fg
    main(bg, fg, args.out_path)
