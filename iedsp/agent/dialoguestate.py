from .ontology import Ontology


class DialogueState(object):
    """
        A multi-framed dialogue state based on Ontology
    """

    def __init__(self):
        self.request_slots = []
        self.inform_slots = []
        self.query_slots = []
        self.mask_str_slots = []

    def add(slot, value, conf):
        pass

    def update_slots(self, slots):
        pass
