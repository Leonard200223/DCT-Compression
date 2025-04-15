import cv2
import numpy as np
from scipy.fftpack import dct, idct
from zigzag import zigzag, inverse_zigzag
import argparse

# =============================
# === CONFIGURABLE PARAMS ====
# =============================
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

# =============================
# === IMAGE UTILITIES ========
# =============================
def rgb_to_ycbcr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def ycbcr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

def downsample(channel):
    return channel[::2, ::2]

# =============================
# === DCT AND BLOCK HANDLING =
# =============================
def block_process(channel, Q):
    h, w = channel.shape
    h8 = h - h % 8
    w8 = w - w % 8
    channel = channel[:h8, :w8]

    blocks = []
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = channel[i:i+8, j:j+8] - 128
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quant_block = np.round(dct_block / Q)
            zz = zigzag(quant_block)
            blocks.append(zz)
    return blocks, h8, w8

# =============================
# === MAIN ENCODER ===========
# =============================
def encode_image(image_path, quality):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycbcr = rgb_to_ycbcr(img)
    
    Y, Cb, Cr = cv2.split(img_ycbcr)
    Cb_down = downsample(Cb)
    Cr_down = downsample(Cr)

    Q = get_quant_matrix(quality)

    Y_blocks, h, w = block_process(Y, Q)
    Cb_blocks, _, _ = block_process(Cb_down, Q)
    Cr_blocks, _, _ = block_process(Cr_down, Q)

    print(f"Image encoded with {len(Y_blocks)} Y blocks, {len(Cb_blocks)} Cb, {len(Cr_blocks)} Cr")

    # Write to file (simplified format)
    with open("compressed_image.txt", "w") as f:
        f.write(f"{h},{w},{quality}\n")
        for block in Y_blocks + Cb_blocks + Cr_blocks:
            f.write(",".join(map(str, block)) + "\n")

# =============================
# === CLI ====================
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image compression using DCT and quantization")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--quality", type=int, default=50, help="Quality factor (1-100)")
    args = parser.parse_args()

    encode_image(args.input, args.quality)
