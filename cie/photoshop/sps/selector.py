import itertools

import cv2
import numpy as np
from skimage.measure import find_contours

from .actions import PSAct, PSArgs
from . import utils


class Selector(object):
    """
    Selector module for SimplePhotoshop
    Takes Photoshop Object as the first argument for every function
    """
    def mask_region(edit_func):
        """
        Uses SimplePhotoshop's mask as selected region
        """
        def wrapper(ps, edit_type, arguments):
            # Get inverse masked image
            has_selection = len(ps.masks) > 0
            if has_selection:  # May contain multiple objects
                mask = np.zeros_like(ps.img)
                for object_name, object_mask in ps.masks:
                    mask = cv2.bitwise_or(mask, object_mask)
                inv_mask = cv2.bitwise_not(mask)
                # Get original image excluding the masked regions
                inv_masked_img = cv2.bitwise_and(inv_mask, ps.img)

            # Edit the whole image
            edited_img = edit_func(ps, edit_type, arguments)
            # Get the masked edited image
            if edited_img is not None and has_selection:
                masked_edited_img = cv2.bitwise_and(
                    mask, edited_img)  # mask the edited image
                edited_img = masked_edited_img + inv_masked_img  # Combine the two
            return edited_img
        return wrapper

    mask_region = staticmethod(mask_region)

    @staticmethod
    def validate_mask(ps, mask):
        """
        Validates mask for Photoshop img
        """
        if ps.img is None:
            return False
        if ps.img.shape != mask.shape:
            return False
        return True

    @staticmethod
    def apply_polygon(image, mask, color=None, caption=None):
        """
        Draw a polygon onto the image 
        """
        # Find contours
        if isinstance(mask, str):
            mask = utils.b64_to_img(mask)
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask[:, :, 0]
        contours = find_contours(padded_mask, 0.5)

        # Plot contours
        masked_image = image.copy()
        for verts in contours:
            # Subtract the padding
            # and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            pts = verts.reshape(-1, 1, 2).astype(np.int32)
            masked_image = cv2.polylines(masked_image, [pts], True, color)

        return masked_image

    @staticmethod
    def apply_caption(image, mask, caption, color):
        captioned_image = image.copy()
        x, y, _ = Selector.find_mask_centroid(mask)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2

        captioned_img = cv2.putText(
            captioned_image, caption, (y, x), fontFace, fontScale, color, thickness)
        return captioned_img

    @staticmethod
    def find_mask_centroid(mask):
        """
        Find the centroid of a 3 dimensional uint8 mask image
        """
        assert ((mask == 0) | (mask == 255)).all() # Either 0 or 255
        X, Y, Z = mask.shape
        moment_x = 0
        moment_y = 0
        moment_z = 0
        npixels = 0
        for x, y, z in itertools.product(range(X), range(Y), range(Z)):
            if mask[x][y][z] == 255:
                moment_x += x
                moment_y += y
                moment_z += z
                npixels += 1
        moment_x /= npixels
        moment_y /= npixels
        moment_z /= npixels
        return round(moment_x), round(moment_y), round(moment_z)
