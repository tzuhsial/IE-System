import copy
import json
import logging
import sys

from .graph import BasicNode
from ..ontology import getOntologyWithName

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IntentStack(object):
    """
    A C++ like stack to store intent slots
    With some convenient functions e.g. prioritize
    """

    def __init__(self):
        self._stack = list()

    def __contains__(self, intent_table):
        return intent_table in self._stack

    def prioritize(self, intent_table):
        """
        Move intent_table to top of stack, 
        Regardless of whether is in stack already
        """
        if intent_table in self._stack:
            self._stack.remove(intent_table)
        self.push(intent_table)

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


class DialogueState(object):
    """
    Multi-domain dialogue state based on Ontology
    Attributes:
        slots (dict): slot -> slot_class mapping
        domains (dict): domain -> intent_class mapping
        intent_stack (list): list of domain
    """

    def __init__(self, config):
        """
        Args:
            ontology
        """
        # Set up with config
        self.ontology = getOntologyWithName(config["ONTOLOGY"])
        self.default_thresh = float(config["DEFAULT_THRESHOLD"])


if __name__ == "__main__":
    pass
