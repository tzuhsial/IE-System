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
    OPEN = "open"
    INFORM = "inform"
    CLOSE = "close"
    AFFIRM = "affirm"
    NEGATE = "negate"

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
        return [UserAct.OPEN, UserAct.INFORM, UserAct.CLOSE]

    @staticmethod
    def photoshop_acts():
        """
        Actions that users directly interact with Photoshop
        """
        return [UserAct.OPEN, UserAct.CLOSE]


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
