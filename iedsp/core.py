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
    User dialogue acts, a.k.a User Intents
    """
    # Domain Acts
    OPEN = "open"
    LOAD = "load"
    ADJUST = "adjust"
    REDO = "redo"
    UNDO = "undo"
    CLOSE = "close"

    # Confirm Acts
    AFFIRM = "affirm"
    NEGATE = "negate"

    @staticmethod
    def confirm_acts():
        """
        Actions that are responses to system's confirm
        """
        return [UserAct.AFFIRM, UserAct.NEGATE]

    @staticmethod
    def domain_acts():
        """
        Actions that are in the image editing domain
        """
        return [UserAct.OPEN, UserAct.LOAD, UserAct.ADJUST, UserAct.REDO, UserAct.UNDO, UserAct.CLOSE]
    
    @staticmethod
    def photoshop_acts():
        """
        Actions that users directly interact with Photoshop
        """
        return [UserAct.OPEN, UserAct.LOAD, UserAct.CLOSE]


class SystemAct:
    """
    System dialogue acts
    """
    GREETING = "greeting"
    ASK = "ask"
    OOD_NL = "ood_nl"
    REQUEST = "request"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    OOD_CV = "ood_cv"
    REPEAT = "repeat"
    REQUEST_LABEL = "request_label"
    REQUEST_CV = "request_cv"
    CONFIRM_CV = "confirm_cv"
    BYE = "bye"


class Hermes:
    """
    Hermes, the emissary of Gods, is a utility class to help
    - user build 1. action 2. goals
    - system build 1. action

    and provides build functions for
    - 1. action
    - 2. slot_dict

    TODO: validate functions
    """
    @staticmethod
    def build_act(dialogue_act, slots=None, speaker=None):
        """
        Act
        {
            'dialogue_act': {
                'value': intent, 'conf': intent_conf,
            },
            'slots': [
                { 'slot': s1, 'value': v1, 'conf': c1 },
                { 'slot': s2, 'value': v2, 'conf': c2 }
            ]
            'speaker': speaker,
        }
        """
        obj = {}
        obj['dialogue_act'] = Hermes.build_slot_dict(
            'dialogue_act', dialogue_act)
        if slots is not None:
            obj['slots'] = slots
        if speaker is not None:
            obj['speaker'] = speaker
        return obj

    @staticmethod
    def build_slot_dict(slot, value=None, conf=None):
        """
        Slot dict format
        {
            'slot' : s1,
            'value': v1,
            'conf' : c1
        }
        """
        obj = {}
        obj['slot'] = slot
        if value is not None:
            obj['value'] = value
        if conf is not None:
            obj['conf']
        return obj
