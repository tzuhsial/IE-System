

class Action(object):
    def __init__(self, dialogue_act, intent, slots):
        self.dialogue_act = dialogue_act
        self.intent = intent
        self.slots = slots