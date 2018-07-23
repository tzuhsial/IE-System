import copy
from collections import defaultdict
from functools import wraps
import json

import cv2
import numpy as np

from .history import EditHistory
from .edit.adjust import adjust
from .edit.select import select
from .state import object_state_factory
from . import utils


class Selector(object):
    """
    Selector module for SimplePhotoshop
    Provides decorators for edit actions
    """
    def mask_region(edit_func):
        """
            Uses SimplePhotoshop's mask as selected region
        """
        def wrapper(ps, edit_type, arguments):
            # If object_mask_id exists
            if "object_mask_id" in arguments:
                object_mask_id = int(arguments['object_mask_id'])
                selected_mask = ps.masks[object_mask_id]
                ps.masks = [selected_mask]

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
            if has_selection:
                masked_edited_img = cv2.bitwise_and(
                    mask, edited_img)  # mask the edited image
                edited_img = masked_edited_img + inv_masked_img  # Combine the two
                arguments['mask'] = mask
            return edited_img
        return wrapper

    mask_region = staticmethod(mask_region)


class SimplePhotoshop(object):
    """
    A photoshop imitation with minimalist features
    Also records history of edits as the same photoshop
    Based on Trung's Annotation Framework https://wiki.corp.adobe.com/display/~bui/Learning+to+map+between+natural+language+requests+to+image+editing+actions
    """

    def __init__(self):
        # Background image
        self.background = None
        self.img = None

        # List of tuples: [(noun1, mask1), (noun2, mask2)...]
        self.masks = list()

        self.history = EditHistory()

        # User and dialogue manager sees this
        self.state = defaultdict(lambda: object_state_factory())

    def reset(self):
        """Resets the image, history and state
        """
        self.background = self.img = None
        self.masks.clear()
        self.history.reset()
        self.state.clear()

    def getState(self):
        """Returns photoshop state, see state.py for more details
        """
        return self.state

    def getImage(self):
        """Returns image with selection mask if present
        """
        img = self.img
        if len(self.getMasks()) > 0:
            masks = self.getMasks()
            colors = utils.random_colors(len(masks))
            for (mask_id, mask), color in zip(masks, colors):
                img = utils.apply_mask(img, mask, color)
        return img

    def getMasks(self):
        return self.masks

    def control(self, control_type, arguments={}):
        """Photoshop control actions
            - open
            - load
            - close
            - undo
            - redo
            - select
            - choose
            - deselect
            - save
        """
        control_msg = "{} : {}".format(control_type, json.dumps(arguments))

        if control_type == "open":
            """Loads and image using image_path argument
            """
            # Load new image
            self.reset()
            image_path = arguments['image_path']
            self.history._background = self.background = self.img = utils.imread(
                image_path)
            self.state['global'] = object_state_factory()
            return True

        elif control_type == "load":
            self.reset()
            b64_img_str = arguments['b64_img_str']
            self.history._background = self.background = self.img = utils.b64_to_img(
                b64_img_str)
            self.state['global'] = object_state_factory()

        elif control_type == "close":
            self.reset()

        elif control_type == "redo":
            if not self.history.hasNextHistory():
                return False
            (action_type, arguments), self.img = self.history.redo()

        elif control_type == "undo":
            if not self.history.hasPreviousHistory():
                return False
            (action_type, arguments), self.img = self.history.undo()

        elif control_type == "select_object_mask_id":
            object_mask_id = int(arguments.get(
                'object_mask_id'))  # Should be m
            self.masks = [self.masks[object_mask_id]]

        elif control_type == "load_masks":
            self.masks.clear()

            mask_strs = arguments.get('masks')
            for mask_idx, mask_str in mask_strs:
                tup = (mask_idx, utils.b64_to_img(mask_str))
                self.masks.append(tup)

        elif control_type == "select_object":
            # API select
            self.masks.clear()
            noun = arguments.get('object')
            masks = select(self.getImage(), noun)
            for mask_idx, mask in enumerate(masks, 0):
                tup = (noun + str(mask_idx), mask)
                self.masks.append(tup)

        elif control_type == "choose":
            mask_indices = arguments['choose_indices']  # Should be a list
            # Mask indices start from 1
            self.masks = [self.masks[idx-1] for idx in mask_indices]

        elif control_type == "deselect":
            self.masks.clear()

        elif control_type == "save":
            self.save()
        else:
            raise ValueError("Unknown control_type: {}".format(control_type))
        return True

    def execute(self, edit_type, arguments):
        """Execute an edit, add to history, update state
        """
        # Edit the image w/o mask
        self.img = self.edit(edit_type, arguments)

        # Update state and add to history
        self.stateUpdate(arguments)
        self.history.add(edit_type, arguments, self.img)

        return True

    @Selector.mask_region
    def edit(self, edit_type, arguments):
        """ Given image edit request, perform edit
            Decorator will apply mask if mask exists
        """
        if edit_type == 'adjust':
            # Edit Image
            attribute = arguments['attribute']
            adjustValue = arguments['adjustValue']
            edited_img = adjust(self.img, attribute, adjustValue)
        else:
            raise ValueError("Unknown edit_type: {}".format(edit_type))

        self.masks.clear()

        return edited_img

    def stateUpdate(self, arguments):
        if len(self.masks) == 0:
            object_names = ['global']
        else:
            object_names = [tup[0] for tup in self.masks]

        for object_name in object_names:
            attribute = arguments.get('attribute')
            adjustValue = arguments.get('adjustValue')
            self.state[object_name][attribute] += adjustValue

    def save(self):
        raise NotImplementedError

    def plot_diff(self, figname=None):
        img1 = self.img
        img2 = self.background
        utils.plot_diff(img1, img2, figname=figname)
