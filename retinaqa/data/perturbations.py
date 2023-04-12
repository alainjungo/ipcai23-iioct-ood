from scipy import ndimage
import numpy as np


def stripes(arr: np.ndarray, min_=2, max_=5, min_int=100, max_int=200):
    arr = arr.copy()
    nb_stripes = np.random.randint(min_, max_ + 1)
    for index in np.random.randint(0, len(arr), nb_stripes):
        arr[index] = np.random.randint(min_int, max_int + 1)
    return arr


def intensity_shift(arr: np.ndarray, min_=25, max_=50, sym=True):
    arr = arr.copy()
    intensity = np.random.randint(min_, max_ + 1)
    if np.random.rand() < 0.5 or not sym:
        arr += intensity
    else:
        arr -= intensity
    arr = np.clip(arr, 0, 255)
    return arr


def contrast(arr: np.ndarray, factors=(0.1, 0.2, 0.3, 2, 3, 4)):
    factor = factors[np.random.randint(0, len(factors))]

    # # same as PIL ImageEnhance.Contrast does
    # # centers at the mean
    # arr = arr * factor + arr.mean() * (1 - factor)

    # we want perturbations, no center at the mean
    arr = arr * factor
    arr = np.clip(arr, 0, 255)

    return arr


def shift(arr: np.ndarray, target: np.ndarray = None, min_=25, max_=100, min_height=10, max_height=80, pos_at_end=False):
    arr = arr.copy()
    if target is not None:
        target = target.copy()

    shift_dist = np.random.randint(min_, max_+1)
    height = np.random.randint(min_height, max_height+1)
    pos = len(arr) - height if pos_at_end else np.random.randint(1, len(arr) - height)   # random range not until end (i.e. len - height + 1) since 2 edges desired

    arr[pos:pos+height] = np.roll(arr[pos:pos+height], shift_dist, axis=-1)
    if target is not None:
        target[pos:pos+height] = np.roll(target[pos:pos+height], shift_dist, axis=-1)
    return arr, target


def shift_multiple(arr: np.ndarray, target: np.ndarray = None, min_=25, max_=100, min_nb=3, max_nb=5):
    arr = arr.copy()
    if target is not None:
        target = target.copy()
    count = np.random.randint(min_nb, max_nb + 1)
    positions = np.sort(np.random.choice(np.arange(1, len(arr) - 1), count, replace=False))
    for i in range(count):
        shift_dist = np.random.randint(min_, max_ + 1)
        if i % 2 == 1:
            shift_dist = -shift_dist
        arr[positions[i]:] = np.roll(arr[positions[i]:], shift_dist, axis=-1)
        if target is not None:
            target[positions[i]:] = np.roll(target[positions[i]:], shift_dist, axis=-1)
    return arr, target


def noise(arr: np.ndarray, sigma=50):
    return arr + np.random.randn(*arr.shape) * sigma


def smooth(arr: np.ndarray, sigma=5):
    # return ndimage.gaussian_filter(arr, sigma)
    return ndimage.gaussian_filter1d(arr, sigma, axis=-1)


def rectangle(arr: np.ndarray, min_width=15, max_width=30, min_height=50, max_height=80, min_int=100, max_intensity=200):
    arr = arr.copy()
    width = np.random.randint(min_width, max_width+1)
    horz_pos = np.random.randint(0, arr.shape[-1] - width)
    height = np.random.randint(min_height, max_height+1)
    vert_pos = np.random.randint(0, len(arr) - height + 1)
    intensity = np.random.randint(min_int, max_intensity+1)
    arr[vert_pos:vert_pos+height, horz_pos:horz_pos+width] = intensity
    return arr


def zoom(arr: np.ndarray, target: np.ndarray = None, min_factor=1.5, max_factor=1.75):
    factor = np.random.rand() * (max_factor - min_factor) + min_factor

    def zoom_with_factor(im):
        zoomed = ndimage.zoom(im, (1, factor))
        mid = zoomed.shape[-1] // 2
        low_half, upper_half = im.shape[-1] // 2 + (im.shape[-1] % 2), im.shape[-1] // 2
        crop_zoomed = zoomed[:, mid - low_half:mid + upper_half]
        return crop_zoomed

    arr_zoomed = zoom_with_factor(arr)
    target_zoomed = zoom_with_factor(target) if target is not None else None
    return arr_zoomed, target_zoomed

