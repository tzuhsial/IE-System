import numpy as np

from . import utils


class EditHistory(object):
    """
        Stores the edit history of Photoshop
        Also stores the resulting image for convenience
    """

    def __init__(self):
        """Use 2 stacks 1. _actions 2. _images to record history
        """
        self._background = None
        self._actions = list()
        self._images = list()
        self._ptr = -1

    def __len__(self):
        assert len(self._actions) == len(self._images)
        return len(self._actions)

    def reset(self):
        self._background = None
        self._actions.clear()
        self._images.clear()
        self._ptr = -1

    def add(self, edit_type, args, img):
        """ Add edit action & result img to history.
            Clears rest of _action, if _ptr not at last pos
        """
        assert self._background is not None, "Need to load image before performing edits!"

        self._actions = self._actions[:self._ptr + 1]
        self._images = self._images[:self._ptr + 1]

        self._actions.append([edit_type, args])
        self._images.append(img)

        self._ptr += 1

    def hasPreviousHistory(self):
        return self._ptr >= 0

    def hasNextHistory(self):
        return self._ptr >= 0 and self._ptr < (len(self._actions) - 1)

    def undo(self):
        assert self._ptr >= 0
        self._ptr -= 1
        if self._ptr >= 0:
            return self._actions[self._ptr], self._images[self._ptr]
        else:  # No actions performed
            return ('none', {}), self._background

    def redo(self):
        assert self._ptr < (len(self._actions) - 1)
        self._ptr += 1
        return self._actions[self._ptr], self._images[self._ptr]

    def to_json(self):
        obj = {}
        if self._background is not None:
            obj["background"] = utils.img_to_b64(self._background)
        obj["actions"] = self._actions
        obj["images"] = [utils.img_to_b64(img) for img in self._images]
        obj["ptr"] = self._ptr
        return obj

    def from_json(self, obj):
        if 'background' in obj:
            self._background = utils.b64_to_img(obj["background"])
        self._actions = obj["actions"]
        self._images = [utils.b64_to_img(i) for i in obj["images"]]
        self._ptr = obj["ptr"]
