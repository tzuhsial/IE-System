import cv2


def crop(img, top, bottom, left, right):
    cropped_img = img[top:bottom, left:right, :]
    return cropped_img
