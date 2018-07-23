

class Message(dict):
    """
    A special type of dict that is used for observation
    """

    def __init__(self):
        pass


def build_act(dialogue_act, intent, slots=None):
    """
    Utility function to build message object that helps communication
    """
    message = {}
    message['dialogue_act'] = dialogue_act
    message['intent'] = intent
    if slots is not None:
        message['slots'] = slots
    return message
