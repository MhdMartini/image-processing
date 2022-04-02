from ast import arg
import numpy as np
import cv2


def linear_gradient(shape, start, end):
    im = np.zeros(shape, dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            d = i + j
            dn = d / sum(shape)
            dnr = 1 - dn
    return


def h_kernel(shape):
    row = np.linspace(0, 1, shape[1])
    kernel_1d = np.tile(row, (shape[0], 1))
    kernel_3d = cv2.merge((kernel_1d, kernel_1d, kernel_1d))
    return kernel_3d


def v_kernel(shape):
    kernel = h_kernel((shape[1], shape[0], shape[2]))
    return cv2.rotate(kernel, cv2.cv2.ROTATE_90_CLOCKWISE)


def d_kernel(shape):
    kernel_h = h_kernel(shape)
    kernel_v = v_kernel(shape)
    kernel_d = (kernel_h + kernel_v) / 2
    return kernel_d


def gradient(shape, start, end, kernel_func):
    im = np.zeros(shape)
    kernel = kernel_func(shape)
    im = kernel * start + (1 - kernel) * end
    return im.astype(np.uint8)


def main(shape, start, end, out_path, kernel_func):
    grad = gradient(shape, start, end, kernel_func)
    cv2.imwrite(out_path, grad)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, nargs="+",
                        default=(255, 255, 255), help="first rgb of gradient")
    parser.add_argument('--end', type=int, nargs="+",
                        default=(47, 32, 9), help="last rgb of gradient")
    parser.add_argument('--gradient', type=int,
                        help="0: horizontal, 1: vertical, 2: diagonal")
    parser.add_argument('--shape', type=int, nargs="+",
                        default=(540, 960, 3), help="shape of 3d output image")
    parser.add_argument('--out_path', type=str, default="./g.png",
                        help="path to output image")
    args = parser.parse_args()

    start = args.start[::-1]
    end = args.end[::-1]
    kernel_funcs = [h_kernel, v_kernel, d_kernel]
    main(args.shape, start, end, args.out_path, kernel_funcs[args.gradient])
