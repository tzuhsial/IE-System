class DialogueState(object):
    """
        Base class of the dialogue state
    """

    def __init__(self):
        raise NotImplementedError

    def add(self, intent, slot, value, conf, turn):
        raise NotImplementedError

    def _build_frame(self, intent, args):
        raise NotImplementedError


class MultiFramedDialogueState(DialogueState):
    """
        Multi-framed dialogue state based on Ontology

        Args:

        Attributes:
        - intentStack (list)
        - frames (dict)
    """

    def __init__(self):
        self.intentStack = list()  # Store user intent as stack
        self.frames = dict()

    def addFrame(self, intent, args=[]):
        """Add frame with intent and arguments
        """
        frame = {}
        for arg in args:
            frame[arg] = {'value': None, 'conf': None, 'turn': 0}
        self.frames[intent] = frame

    def add(self, intent, slot, value, conf, turn):
        """ Add intent, slot, value, conf, turn info to dialogue state

            Args:
            - intent (str)
            - slot (str)
            - value (str/int/float)
            - conf (float)
            - turn (int)

            Returns:
            - status (bool)
        """
        if intent not in self.frames:
            return False

        if len(self.intentStack) == 0 or intent != self.intentStack[-1]:
            self.intentStack.append(intent)
        currIntent = self.intentStack[-1]

        if slot not in self.frames[currIntent]:
            return False

        currFrame = self.frames[currIntent]

        currFrame['value'] = value
        currFrame['conf'] = conf
        currFrame['turn'] = turn

        self.frames[currIntent] = currFrame

        return True

    def status(self):
        """Returns current status according to frames
        """
        pass


if __name__ == "__main__":

    state = MultiFramedDialogueState()
    state.addFrame('open', ['image_path'])
    state.addFrame('adjust', ['attribute', 'adjustValue', 'object'])
    state.addFrame('undo')

    turn = 1

    state.add('adjust', 'attribute', 'brightness')

    import pdb
    pdb.set_trace()
    pass
