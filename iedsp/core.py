import json
from .util import sort_slots_with_key


class Agent:
    """
    Agents participating in the self play world
    """
    USER = "user"
    CHANNEL = "channel"
    SYSTEM = "system"
    PHOTOSHOP = "photoshop"


class UserAct:
    """
    User Dialogue Acts
    """
    INFORM = "inform"
    AFFIRM = "affirm"
    NEGATE = "negate"
    WAIT = "wait"

    @staticmethod
    def confirm_acts():
        """
        Actions that are responses to system's confirm
        """
        return [UserAct.AFFIRM, UserAct.NEGATE]

    @staticmethod
    def intent_acts():
        """
        Actions that are in the image editing domain
        """
        return [UserAct.INFORM]

    @staticmethod
    def wait_acts():
        """
        Wait for the system to query...
        """
        return[UserAct.WAIT]


class SystemAct:
    """
    System dialogue acts
    """
    GREETING = "greeting"
    ASK = "ask"
    REQUEST = "request"
    CONFIRM = "confirm"
    QUERY = "query"
    QUERY_VISIONENGINE = "query_visionengine"
    QUERY_HISTORY = "query_history"
    EXECUTE = "execute"
    BYE = "bye"

    @staticmethod
    def load_mask_strs_acts():
        """
        Actions that photoshop needs to load mask strs
        """
        return [SystemAct.REQUEST, SystemAct.CONFIRM, SystemAct.EXECUTE]

    @staticmethod
    def query_acts():
        """
        Actions that requires the user to wait
        """
        return [SystemAct.QUERY, SystemAct.QUERY_VISIONENGINE, SystemAct.QUERY_HISTORY]


class SysIntent(object):
    """
    A utility object for system intent slots
    """

    def __init__(self, confirm_slots=None, request_slots=None, execute_slots=None):
        """
        default argument cannot be list -> a bug that fucked me for 2 days...
        """
        if confirm_slots is None:
            confirm_slots = []
        if request_slots is None:
            request_slots = []
        if execute_slots is None:
            execute_slots = []
        self.confirm_slots = confirm_slots
        self.request_slots = request_slots
        self.execute_slots = execute_slots

    def __iadd__(self, other_intent):
        self.confirm_slots += other_intent.confirm_slots
        self.request_slots += other_intent.request_slots
        self.execute_slots += other_intent.execute_slots
        return self

    def __add__(self, other_intent):
        confirm_slots = self.confirm_slots + other_intent.confirm_slots
        request_slots = self.request_slots + other_intent.request_slots
        execute_slots = self.execute_slots + other_intent.execute_slots
        return SysIntent(confirm_slots, request_slots, execute_slots)

    def __radd__(self, other_intent):
        return other_intent.__add__(self)

    def __eq__(self, other_intent):
        """
        Compare slots, regardless of order
        """
        if sort_slots_with_key('slot', self.confirm_slots) != sort_slots_with_key('slot', other_intent.confirm_slots):
            return False
        if sort_slots_with_key('slot', self.request_slots) != sort_slots_with_key('slot', other_intent.request_slots):
            return False
        if sort_slots_with_key('slot', self.execute_slots) != sort_slots_with_key('slot', other_intent.execute_slots):
            return False
        return True

    def empty(self):
        if len(self.confirm_slots) != 0:
            return False
        elif len(self.request_slots) != 0:
            return False
        elif len(self.execute_slots) != 0:
            return False
        return True

    def clear(self):
        self.confirm_slots.clear()
        self.request_slots.clear()
        self.execute_slots.clear()

    def copy(self):
        return SysIntent(self.confirm_slots, self.request_slots, self.execute_slots)

    def to_json(self):
        obj = {
            'confirm': self.confirm_slots,
            'request': self.request_slots,
            'execute': self.execute_slots
        }
        return obj

    def pprint(self):
        print('confirm', self.confirm_slots)
        print('request', self.request_slots)
        print('execute', self.execute_slots)

    def executable(self):
        """
        Returns True iff all slots are executable
        """
        if len(self.confirm_slots) != 0:
            return False
        if len(self.request_slots) != 0:
            return False
        if len(self.execute_slots) != 0:
            return True
        return False


class PhotoshopAct:
    """
    Defines actions that are supported by photoshop
    """
    OPEN = "open"
    LOAD = "load"
    CLOSE = "close"
    REDO = "redo"
    UNDO = "undo"
    LOAD_MASK_STRS = "load_mask_strs"
    SELECT_OBJECT = "select_object"
    SELECT_OBJECT_MASK_ID = "select_object_mask_id"
    DESELECT = "deselect"

    ADJUST = "adjust"
    ADJUST_COLOR = "adjust_color"

    @staticmethod
    def control_acts():
        return [PhotoshopAct.OPEN, PhotoshopAct.LOAD, PhotoshopAct.CLOSE, PhotoshopAct.UNDO,
                PhotoshopAct.REDO, PhotoshopAct.LOAD_MASK_STRS,
                PhotoshopAct.SELECT_OBJECT, PhotoshopAct.SELECT_OBJECT_MASK_ID, PhotoshopAct.DESELECT]

    @staticmethod
    def edit_acts():
        return [PhotoshopAct.ADJUST, PhotoshopAct.ADJUST_COLOR]
