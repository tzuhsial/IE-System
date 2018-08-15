import copy
import logging

from ..core import UserAct, SysIntent
from ..ontology import OntologyPortal
from ..util import slot_to_observation

logger = logging.getLogger(__name__)


def StatePortal(global_config):
    """
    Constructs state with global_config
    """
    ontology = OntologyPortal(global_config)
    state = State(ontology)
    return state


class ExecutionHistory(object):
    """
    A C++ like stack to store current intent slots
    Also provides search functions like visionengine
    """

    def __init__(self):
        self._stack = list()

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

    def clear(self):
        return self._stack.clear()

    def select_object(self, object, **kwargs):
        """
        Args
            object (str): name of object
        Returns:
            mask_strs (list): list of mask strings
        """
        if object == "image":
            return []

        mask_strs = []
        for intent_tree in self._stack:
            node_dict = intent_tree.node_dict
            if object == node_dict['object'].get_max_value():
                object_mask_str = node_dict['object_mask_str'].get_max_value()
                mask_strs.append(object_mask_str)

        # Remove duplicates
        mask_strs = list(set(mask_strs))
        return mask_strs


class State(object):
    """
    Dialogue State
    Manages intent pulling and frame stacking

    Attributes:
        ontology (object): object that creates intent trees
        framestack (list): frame stack that records previous intent
    """

    def __init__(self, ontology):

        self.ontology = ontology
        self.executionhistory = ExecutionHistory()

        self.sysintent = SysIntent()

    def update(self, dialogue_act, intent, slots, turn_id):
        """
        Args:
            dialogue_act (dict): slot dict
            intent (dict): slot dict
            slots (list): list of slot dict
            turn_id (int): turn index
        """
        self.update_dialogueact(dialogue_act, turn_id)

        self.update_intent(intent, turn_id)

        self.update_slots(slots, turn_id)

    def update_dialogueact(self, dialogue_act, turn_id):
        """
        Mainly handle affirm/negate acts
        Args: 
            dialogue act: slot dict
        """
        if dialogue_act is None or len(dialogue_act) == 0:
            return

        if dialogue_act['value'] in UserAct.confirm_acts():
            value = dialogue_act['value']
            if value == UserAct.AFFIRM:
                conf = dialogue_act['conf']
            elif value == UserAct.NEGATE:
                conf = 1 - dialogue_act['conf']

            # Get previous confirmed slot
            prev_confirm_slot = self.sysintent.confirm_slots[0]

            slot_name = prev_confirm_slot['slot']

            obsrv = slot_to_observation(prev_confirm_slot, turn_id)
            obsrv["conf"] = conf

            self.ontology.get_slot(slot_name).add_observation(**obsrv)

    def update_intent(self, intent_slot, turn_id):
        """
        In other words, update framestack
        Args:
            intent_slot (dict): slot dict 
            turn_id (int): turn index
        """
        if intent_slot is None or len(intent_slot) == 0:
            return
        obsrv = slot_to_observation(intent_slot, turn_id)
        self.ontology.get_slot("intent").add_observation(**obsrv)

    def update_slots(self, slots, turn_id):
        """
        Args:
            slots (list): list of slot dicts
            turn_id (int)
        """
        if slots is None or len(slots) == 0:
            return

        for slot in slots:
            slot_name = slot['slot']
            if slot_name not in self.ontology.slots:
                logger.info("Unknown slot: {}".format(slot_name))
                continue

            obsrv = slot_to_observation(slot, turn_id)
            slot_node = self.get_slot(slot_name)
            result = slot_node.add_observation(**obsrv)
            if not result:
                logger.info(
                    "Failed to add observation {} to slot {}".format(obsrv, slot_name))

    def pull(self):
        """
        Returns the current sysintent of the state
        Checks intent tree first
        """
        sysintent = self.get_intent('intent').pull()
        if sysintent.executable():
            execute_intent = sysintent.execute_slots[0]['value']
            sysintent = self.get_intent(execute_intent).pull()
        self.sysintent = sysintent
        return self.sysintent

    def stack_intent(self, intent_name):
        """
        Pushes the intent with previous values into the intent stack
        """
        intent_tree = self.get_intent(intent_name)
        copied_tree = copy.deepcopy(intent_tree)  # Copy the intent tree
        self.executionhistory.push(copied_tree)

    def get_slot(self, slot_name):
        return self.ontology.get_slot(slot_name)

    def get_intent(self, intent_name):
        return self.ontology.get_intent(intent_name)

    def clear_intent(self, intent_name):
        self.ontology.get_intent(intent_name).clear()

    def clear_slot(self, slot_name):
        self.ontology.get_slot(slot_name).clear()

    def clear_ontology(self):
        self.ontology.clear()

    def clear_history(self):
        self.executionhistory.clear()

    def clear(self):
        self.ontology.clear()
        self.executionhistory.clear()

    def to_json(self):
        """
        Format to json to save as history
        """
        raise NotImplementedError

    def to_list(self):
        """
        State as list for RL agent observation space
        """
        # Slots
        l = []
        for slot in self.ontology.slots.values():
            l += slot.to_list()
        return l
