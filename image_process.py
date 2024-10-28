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