import cv2
import numpy as np

def split_img_horizonal(image: np.ndarray):
    height, _ = image.shape[:2]

    upper = image[: height // 2, :]
    lower = image[height // 2 :, :]

    return upper, lower

def split_img_vertical(image: np.ndarray):
    _, width = image.shape[:2]

    left = image[:, : width // 2]
    right = image[:, width // 2 :]

    return left, right

def calc_color_ratio(img, lower_bound, upper_bound) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    pixels = np.count_nonzero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    return pixels / total_pixels

def crop_image(img, x, y, width, height):
    cropped_img = img[y : y + height, x : x + width]
    return cropped_img

def is_contain_template(src: np.ndarray, template: np.ndarray, threshold=0.7, use_gray=True, x=None, y=None, width=None, height=None):
    if use_gray:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if len(src.shape) == 3 else src
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    if all(v is not None for v in [x, y, width, height]):
        src = src[y : y + height, x : x + width]
    res = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    print(max_val)
    return max_val > threshold