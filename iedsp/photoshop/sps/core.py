import copy
from collections import defaultdict
import itertools
import json

import cv2
import numpy as np
from skimage.measure import find_contours

from .actions import PSAct, PSArgs
from .cvengine import CVEngineClient
from .edit import adjust, adjust_color
from .history import EditHistory
from .selector import Selector
from .state import object_state_factory
from . import utils


class SimplePhotoshop(object):
    """
    A photoshop imitation with minimalist features
    Also records history of edits as the same photoshop
    Based on Trung's Annotation Framework https://wiki.corp.adobe.com/display/~bui/Learning+to+map+between+natural+language+requests+to+image+editing+actions
    """

    def __init__(self, **kwargs):
        # Background image
        self.background = None
        self.img = None

        # CVEngineClient
        self.cvengine = CVEngineClient()

        # List of tuples: [(noun1, mask1), (noun2, mask2)...]
        self.masks = list()

        self.history = EditHistory()

        # User and dialogue manager sees this
        self.state = defaultdict(lambda: object_state_factory())

        # Verbose
        self.verbose = kwargs.get("verbose", False)

    def reset(self):
        """Resets the image, history and state
        """
        self.background = self.img = None
        self.masks.clear()
        self.history.reset()
        self.state.clear()

    def get_state(self):
        """Returns photoshop state, see state.py for more details
        """
        return self.state

    def get_image(self, plot_mask=True):
        """Returns image with selection mask & id if present
        """
        img = self.img

        if plot_mask and len(self.get_masks()) > 0:
            masks = self.get_masks()
            colors = utils.random_colors(len(masks))
            for (mask_id, mask), color in zip(masks, colors):
                img = Selector.apply_polygon(img, mask, color)
                img = Selector.apply_caption(img, mask, str(mask_id), color)
        return img

    def get_masks(self):
        return self.masks

    ###################
    #     Control     #
    ###################
    def control(self, control_type, arguments={}):
        """
        Photoshop Control Actions
        TODO: customize error message
        Args: 
            control_type (str):
            arguments (dict):
        Returns:
            result (bool): True if success, else False
            message(str): execute_message if success, else error_message
        """
        if control_type == PSAct.Control.OPEN:
            result, msg = self.control_open(arguments)
        elif control_type == PSAct.Control.LOAD:
            result, msg = self.control_load(arguments)
        elif control_type == PSAct.Control.CLOSE:
            result, msg = self.control_close()
        elif control_type == PSAct.Control.REDO:
            result, msg = self.control_redo()
        elif control_type == PSAct.Control.UNDO:
            result, msg = self.control_undo()
        elif control_type == PSAct.Control.SELECT_OBJECT:
            result, msg = self.control_select_object(arguments)
        elif control_type == PSAct.Control.SELECT_OBJECT_MASK_ID:
            result, msg = self.control_select_object_mask_id(arguments)
        elif control_type == PSAct.Control.LOAD_MASK_STRS:
            result, msg = self.control_load_mask_strs(arguments)
        elif control_type == PSAct.Control.DESELECT:
            result, msg = self.control_deselect()
        else:
            result = False,
            msg = "failure"
        return result, msg

    def control_open(self, arguments):
        try:
            image_path = arguments.get(PSArgs.IMAGE_PATH, "QQ.jpg")
            img = utils.imread(image_path)
            self.reset()
            self.history._background = self.background = self.img = img
            self.state['global'] = object_state_factory()
            result = True
            message = "success"
        except Exception as e:
            result = False
            message = e
        finally:
            return result, message

    def control_load(self, arguments):
        try:
            b64_img_str = arguments.get(PSArgs.B64_IMG_STR)
            img = utils.b64_to_img(b64_img_str)
            self.reset()
            self.history._background = self.background = self.img = img
            self.state['global'] = object_state_factory()
            result = True
            message = "success"
        except Exception as e:
            result = False
            message = e
        finally:
            return result, message

    def control_close(self, arguments={}):
        self.reset()
        return True, "success"

    def control_redo(self, arguments={}):
        if not self.history.hasNextHistory():
            #print("[photoshop] execution failure: no next history")
            msg = "Error occured when executing \"redo\": no next history"
            return False, msg
        (action_type, arguments), self.img = self.history.redo()
        return True, "success"

    def control_undo(self, arguments={}):
        if not self.history.hasPreviousHistory():
            #print("[photoshop] execution failure: no previous history")
            msg = "Error occured when executing \"undo\": no previous history"
            return False, msg
        (action_type, arguments), self.img = self.history.undo()
        return True, "success"

    def control_select_object(self, arguments):
        try:
             # API select
            noun = arguments.get('object')
            masks = []

            mask_arrs = self.cvengine.select(self.img, noun)

            for mask_idx, mask in enumerate(mask_arrs, 0):
                tup = (str(mask_idx), mask)
                masks.append(tup)
            self.masks = masks
            result = True
            message = 'success'
        except Exception as e:
            result = False
            message = e
        finally:
            return result, message

    def control_select_object_mask_id(self, arguments):
        try:
            object_mask_id = str(arguments.get(
                PSArgs.OBJECT_MASK_ID))  # Should be m

            mask = None
            for mask_tuple in self.masks:
                if mask_tuple[0] == object_mask_id:  # str
                    mask = mask_tuple[1]
                    break

            if mask is None:
                result = False
                message = "object_mask_id {} not found!".format(object_mask_id)
                raise ValueError

            self.masks = [("0", mask)]
            result = True
            message = "success"
        except Exception as e:

            result = False
            message = e
        finally:
            return result, message

    def control_load_mask_strs(self, arguments):
        try:
            mask_strs = arguments.get(
                PSArgs.MASK_STRS, list())  # list of b64_img_str

            masks = []
            for mask_idx, mask_str in mask_strs:
                mask = utils.b64_to_img(mask_str)
                if not Selector.validate_mask(self, mask):
                    print("LOAD_MASK_STRS: Invalid mask!")
                    raise ValueError
                tup = (mask_idx, mask)
                masks.append(tup)
            self.masks = masks
            result = True
            message = "success"
        except Exception as e:
            result = False
            message = e
        finally:
            return result, message

    def control_deselect(self, arguments={}):
        if len(self.masks):
            self.masks.clear()
            return True, "success"
        else:
            return False, "failure"

    ######################
    #       Edit         #
    ######################
    def execute(self, edit_type, arguments):
        """
        1. Execute an editA
        2. Add to history
        3. Update state
        4. Clear masks
        """
        # Edit the image w/o mask
        edited_img = self.edit(edit_type, arguments)

        if edited_img is not None:
            result = True
            msg = "success"
        else:
            result = False
            msg = "failure"

        # Update state and add to history
        if result is True:
            self.img = edited_img
            self.state_update(edit_type, arguments)
            self.history.add(edit_type, arguments, self.img)
            # self.masks.clear()
        return result, msg

    @Selector.mask_region
    def edit(self, edit_type, arguments):
        """ 
        Args:
            edit_type (str)
            arguments (dict)
        Returns:
            result (bool)
            msg (str)
        """
        if edit_type == PSAct.Edit.ADJUST:
            edited_img = self.edit_adjust(arguments)
        elif edit_type == PSAct.Edit.ADJUST_COLOR:
            edited_img = self.edit_adjust_color(arguments)
        else:
            #print("Unknown edit_type: {}".format(edit_type))
            edited_img = None
        return edited_img

    def edit_adjust(self, arguments):
        try:
            attribute = arguments.get(PSArgs.ATTRIBUTE)
            adjust_value = arguments.get(PSArgs.ADJUST_VALUE)
            edited_img = adjust(self.img, attribute, adjust_value)
        except Exception as e:
            edited_img = None
        return edited_img

    def edit_adjust_color(self, arguments):
        try:
            color = arguments.get(PSArgs.COLOR)
            edited_img = adjust_color(self.img, color)
        except Exception as e:
            edited_img = None
        return edited_img

    def state_update(self, edit_type, arguments):
        if len(self.masks) == 0:
            object_names = ['global']
        else:
            object_names = [tup[0] for tup in self.masks]

        #print('object_names', object_names)
        #print('arguments', arguments)

        for object_name in object_names:
            if edit_type == PSAct.Edit.ADJUST:
                attribute = arguments.get('attribute')
                adjust_value = arguments.get('adjust_value')
                self.state[object_name][attribute] += adjust_value
            elif edit_type == PSAct.Edit.ADJUST_COLOR:
                color = arguments.get('color')
                self.state[object_name]['color'] = color

    def plot_diff(self, figname=None):
        img1 = self.img
        img2 = self.background
        utils.plot_diff(img1, img2, figname=figname)
