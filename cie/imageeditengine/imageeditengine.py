import colorsys
import logging

import numpy as np
from skimage.measure import find_contours

import cv2

from .. import util
from ..core import SystemAct, UserAct
from .adjust import adjust

logger = logging.getLogger(__name__)


def ImageEditEnginePortal(imageeditengine_config):
    imageeditengine_type = imageeditengine_config["imageeditengine"]
    if imageeditengine_type == "SimpleImageEditEngine":
        return SimpleImageEditEngine()
    else:
        raise NotImplementedError(imageeditengine_type)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    for idx, tup in enumerate(colors):
        colors[idx] = list(map(lambda c: c * 255, tup))
    return colors


def apply_polygon(image, mask, color=None, caption=None):
    """
    Draw a polygon onto the image 
    Args:
        image (np.array)
        mask  (np.array)
    """
    # Find contours
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
        masked_image = cv2.polylines(
            masked_image, pts=[pts], isClosed=True, color=(57, 255, 20), thickness=2)

    return masked_image


class SimpleImageEditEngine(object):
    """
    Our image edit engine only has 1 layer
    Args:
        background (np.array): the background image
        image (np.array): the displayed image
        selector (np.array): mask of the selected region
        state (dict): adjustments
    """

    def __init__(self):
        self.background = None
        self.image = None
        self.mask = None
        self.steps = []  # Empty everything

    def reset(self):
        """
        Reset and start anew
        """
        self.observation = {}
        self.background = None
        self.image = None
        self.mask = None
        self.steps = []  # Empty everything

    def open(self, image_path):
        """
        Resets engine state and reads image
        """
        # Reset
        self.reset()

        # Read image
        self.background = self.image = util.imread(image_path)

    def load_mask(self, b64_mask_str):
        """
        Load mask
        """
        self.mask = util.b64_to_img(b64_mask_str)

    def observe(self, observation):
        self.observation = observation

    def act(self):
        # Get all system actions
        system_acts = self.observation.get('system_acts', list())

        if len(system_acts) > 0:
            sys_act = system_acts[0]
            load_mask_act = system_acts[1]
            sys_dialogue_act = sys_act['dialogue_act']['value']

            if sys_dialogue_act == SystemAct.EXECUTE:
                intent = sys_act['intent']['value']
                slots = sys_act['slots']

                # Now we only support adjust
                edit_args = util.slots_to_args(slots)
                self.edit(intent, edit_args)
                # Remove mask after edit
                self.mask = None
            elif len(load_mask_act['slots']) > 0:
                mask_slot = load_mask_act['slots'][0]
                b64_mask_str = mask_slot.get('value', None)
                if b64_mask_str is not None:
                    mask = util.b64_to_img(b64_mask_str)
                else:
                    mask = None
                self.mask = mask

        ie_act = self.act_inform()
        imageditengine_act = {
            'imageeditengine_acts': [ie_act]
        }
        return imageditengine_act

    def act_inform(self):
        """
        Informs user of action
        """
        if self.background is None:
            original_b64_img_str = ""
        else:
            original_b64_img_str = util.img_to_b64(self.background)

        original_slot = util.build_slot_dict(
            'original_b64_img_str', original_b64_img_str, 1.0)

        slots = [original_slot]

        ie_act = {
            'dialogue_act': util.build_slot_dict('dialogue_act', UserAct.INFORM, 1.0),
            'slots': slots
        }
        return ie_act

    def edit(self, intent, edit_args):
        """
        Adds another edit to the current list of edits
        """
        if intent != "adjust":
            raise NotImplementedError("Imageedit Engine only supports ADJUST!")

        # Add current edit_args to the list of steps
        self.steps.append(edit_args)

        image = self.background.copy()

        for step in self.steps:
            object_mask_str = step['object_mask_str']
            object_mask = util.b64_to_img(object_mask_str)
            attribute = step['attribute']
            adjust_value = step['adjust_value']

            # Create inverse mask image
            inv_mask = cv2.bitwise_not(object_mask)
            inv_masked_image = cv2.bitwise_and(inv_mask, image)

            edited_image = adjust(image, attribute, adjust_value)

            # Add inverse mask image back
            masked_edited_image = cv2.bitwise_and(object_mask, edited_image)
            image = masked_edited_image + inv_masked_image

        # Store image
        self.image = image

    def get_image(self, plot_mask=True):
        """Returns image with mask if present
        """
        image = self.image
        if plot_mask and self.mask is not None:
            colors = random_colors(1)
            image = apply_polygon(self.image, self.mask, colors[0])
        return image

    def to_json(self):
        obj = {
            'b64_background_str': None,
            'b64_image_str': None,
            'b64_mask_str': None,
            'steps': self.steps
        }

        if self.background is not None:
            obj['b64_background_str'] = util.img_to_b64(self.background)
            obj['b64_image_str'] = util.img_to_b64(self.image)

        if self.mask is not None:
            obj['b64_mask_str'] = util.img_to_b64(self.mask)

        return obj

    def from_json(self, obj):
        if obj['b64_background_str'] is not None:
            self.background = util.b64_to_img(obj['b64_background_str'])
            self.image = util.b64_to_img(obj['b64_image_str'])

        if obj['b64_mask_str'] is not None:
            self.mask = util.b64_to_img(obj['b64_mask_str'])

        self.steps = obj['steps']
