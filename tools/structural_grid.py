"""Project scalar self-similarity onto the existing detector's spatial grid."""
import cv2
import numpy as np

from tools.structural_correspondence import descriptor


def structural_grid(image, input_hw):
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Use the original scalar uint8 view')
    height, width = map(int, input_hw)
    if height % 32 or width % 32:
        raise ValueError('Keep all three original pyramid grids integral')
    canvas_hw = (height//2, width//2)
    ratio = min(height/image.shape[0], width/image.shape[1])/2
    h, w = int(image.shape[0]*ratio), int(image.shape[1]*ratio)
    canvas = np.full(canvas_hw, float(image.mean())/255, np.float32)
    canvas[:h, :w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    values, variance = descriptor(canvas)
    yy, xx = np.indices(canvas_hw)
    valid = (yy >= 3) & (xx >= 3) & (yy < h-3) & (xx < w-3) & (variance > 1e-6)
    values = cv2.resize(values, (width//8, height//8), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid.astype(np.float32), (width//8, height//8), interpolation=cv2.INTER_AREA) >= .5
    return values.transpose(2, 0, 1).astype(np.float16), valid[None]
