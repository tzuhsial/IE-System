import cv2


def rotate(img, degree):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))
    return rotated_img