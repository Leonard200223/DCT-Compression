import cv2
import numpy as np
from scipy.fftpack import idct
from zigzag import inverse_zigzag
import argparse


def upsample(channel, h, w):
    return cv2.resize(channel, (w, h), interpolation=cv2.INTER_LINEAR)

def ycbcr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

def get_quant_matrix(quality):
    Q50 = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

    if quality < 50 and quality > 1:
        scale = 5000 / quality
    elif quality < 100:
        scale = 200 - 2 * quality
    else:
        scale = 1

    Q = np.floor((Q50 * scale + 50) / 100)
    Q[Q == 0] = 1
    return Q


def block_reconstruct(blocks, h, w, Q):
    channel = np.zeros((h, w))
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = inverse_zigzag(blocks[idx], 8, 8)
            dequant = block * Q
            idct_block = idct(idct(dequant.T, norm='ortho').T, norm='ortho')
            channel[i:i+8, j:j+8] = idct_block + 128
            idx += 1
    return np.clip(channel, 0, 255).astype(np.uint8)


def decode_image(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    h, w, quality = map(int, lines[0].strip().split(","))
    blocks_raw = [list(map(int, line.strip().split(","))) for line in lines[1:]]

    Q = get_quant_matrix(quality)

    y_blocks = blocks_raw[:(h//8)*(w//8)]
    cb_blocks = blocks_raw[(h//8)*(w//8):(h//8)*(w//8)//4*5]  # 1/4 dimensiune cb+cr
    cr_blocks = blocks_raw[(h//8)*(w//8)//4*5:]

    Y = block_reconstruct(y_blocks, h, w, Q)
    Cb = block_reconstruct(cb_blocks, h//2, w//2, Q)
    Cr = block_reconstruct(cr_blocks, h//2, w//2, Q)

    Cb_up = upsample(Cb, h, w)
    Cr_up = upsample(Cr, h, w)

    img_ycbcr = cv2.merge([Y, Cb_up, Cr_up])
    img_rgb = ycbcr_to_rgb(img_ycbcr)

    cv2.imwrite("decompressed_image.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print("Image reconstructed and saved as decompressed_image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode image from DCT-compressed text")
    parser.add_argument("--input", type=str, required=True, help="Path to compressed image text")
    args = parser.parse_args()

    decode_image(args.input)
