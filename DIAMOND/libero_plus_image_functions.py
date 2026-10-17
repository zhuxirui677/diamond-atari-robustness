# Functions extracted verbatim from sylvestf/LIBERO-plus env_wrapper.py.
# See libero_plus_provenance.json for exact revision and upstream hash.
import numpy as np
from io import BytesIO
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom

def enable_motion_blur():
    global MotionImage, cv2
    import ctypes
    import cv2
    from wand.api import library
    from wand.image import Image
    library.MagickMotionBlurImage.argtypes = (ctypes.c_void_p, ctypes.c_double,
                                               ctypes.c_double, ctypes.c_double)
    class MotionImage(Image):
        def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
            library.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def motion_blur(x, severity=1):
    # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    # c = [(15, 5), (20, 8), (25, 12), (30, 15), (35, 20)][severity - 1]
    c = [
        (5, 2),
        (8, 3),
        (10, 4),
        (12, 5),
        (15, 6),
        (18, 8),
        (20, 10),
        (25, 12),
        (30, 15),
        (35, 20)
    ][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                    cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03),
         np.arange(1, 1.36, 0.01),
         np.arange(1, 1.41, 0.01),
         np.arange(1, 1.46, 0.02),
         np.arange(1, 1.51, 0.02),
         np.arange(1, 1.56, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255

def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def fog(x, severity=1):
    # c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    c = [(0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2), (2.5, 2.0), (3.0, 1.8), (3.5, 1.6), (4.0, 1.5), (4.5, 1.4), (5.0, 1.3)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    height_x, weight_x = x.shape[0], x.shape[1]
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:height_x, :weight_x][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    # c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    c = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2), (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)

    height_x, weight_x = x.shape[0], x.shape[1]

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(height_x - c[1], c[1], -1):
            for w in range(weight_x - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255
