import copy
import logging

from ..core import UserAct
from ..util import slot_to_observation

logger = logging.getLogger(__name__)


class FrameStack(object):
    """
    A C++ like stack to store intent slots
    Also provides some convenient functions e.g. prioritize
    """

    def __init__(self):
        self._stack = list()

    def __contains__(self, intent_tree):
        return intent_tree in self._stack

    def top(self):
        if len(self._stack):
            return self._stack[0]
        return None

    def push(self, intent_tree):
        self._stack.insert(0, intent_tree)

    def pop(self):
        self._stack.pop(0)

    def size(self):
        return len(self._stack)

    def empty(self):
        return self.size() == 0


class State(object):
    """
    Dialogue State
    Manages intent pulling and memory stacking

    Attributes:
        ontology (object): object that creates intent trees
        framestack (list): frame stack that records previous intent
    """

    def __init__(self, ontology):
        self.ontology = ontology
        self.framestack = FrameStack()

        self.intent_tree = self.ontology.create(intent='unknown')

    def update(self, dialogue_act, intent, slots, turn_id):
        """
        Args:
            dialogue_act (dict): slot dict
            intent (dict): slot dict
            slots (list): list of slot dict
            turn_id (int): turn index
        """

        self.update_dialogueact(dialogue_act)

        self.update_intent(intent)

        self.update_slots(slots, turn_id)

    def update_dialogueact(self, dialogue_act):
        """
        Args: 
            dialogue act: slot dict
        """
        if dialogue_act['slot'] in UserAct.confirm_acts:
            raise NotImplementedError

    def update_intent(self, intent_slot):
        """
        In other words, update framestack
        TODO: update according to intent
        Args:
            intent_slot (dict): slot dict 
        """
        unknown_keyword = self.ontology.Intents.UNKNOWN
        intent_keyword = self.ontology.Slots.INTENT
        intent_thresh = self.ontology.thresh.get(
            intent_keyword.upper() + "_THRESHOLD")

        # Initialize with an unknown tree
        if self.framestack.empty():
            unknown_tree = self.ontology.create(unknown_keyword)
            self.framestack.push(unknown_tree)

        # Update current intent
        if self.framestack.top().name == unknown_keyword:
            # Check confidence of current intent

        else:  # Has some undone intent

        pass

    def update_slots(self, slots, turn_id):
        """
        Args:
            slots (list): list of slot dicts
            turn_id (int)
        """
        intent_tree = self.framestack.top()

        for slot in slots:
            slot_name = slot['slot']
            if slot_name in intent_tree.node_dict:
                obsrv = slot_to_observation(slot, turn_id)
                intent_tree.node_dict[slot_name].add_observation(**obsrv)

    def to_json(self):
        """
        Format to json object to save as history
        """
        raise NotImplementedError

    def to_vector(self):
        """
        State as vector for RL observation space
        """
        raise NotImplementedError
