import logging

from ..core import UserAct, SysIntent
from .ontology import OntologyEngine
from .node import builder as nodelib
from ..util import slot_to_observation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class State(object):
    """
    Dialogue State

    Attributes:
        ontology (object): object that creates intent trees
        executionhistory (object): frame stack that records previous intent
        sysintent (object): convient wrapper class
    """

    def __init__(self, ontology_json):
        self.ontology = OntologyEngine(ontology_json)
        self.sysintent = SysIntent()
        self.execution = []

    def reset(self):
        self.ontology.clear()
        self.sysintent.clear()
        self.turn_id = 0

    def flush(self):
        """
        Flushes all slot values in ontology.
        Records the flushed slot values in execution
        """
        exec_state = self.ontology.to_json()
        self.execution.append(exec_state)
        self.ontology.flush()

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
        if dialogue_act is None or\
                len(dialogue_act) == 0 or \
                dialogue_act['conf'] < 0.5:
            return

        # Handle affirm/negate
        if dialogue_act['value'] in UserAct.confirm_acts():
            # Get previous confirmed slot
            if len(self.sysintent.confirm_slots) == 0:
                return

            prev_confirm_slot = self.sysintent.confirm_slots[0]
            slot_name = prev_confirm_slot['slot']
            slot_node = self.ontology.get_slot(slot_name)

            value = dialogue_act['value']

            if issubclass(slot_node.__class__, nodelib('PSToolNode')):
                if value == UserAct.AFFIRM:
                    pass  # Don't have to do anything to the PSToolNode
                else:
                    slot_node.clear()
            else:
                if value == UserAct.AFFIRM:
                    conf = dialogue_act['conf']
                elif value == UserAct.NEGATE:
                    conf = 1 - dialogue_act['conf']

                obsrv = slot_to_observation(prev_confirm_slot, turn_id)
                obsrv["conf"] = conf

                slot_node.add_observation(**obsrv)

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
                logger.debug("Failed to add observation {} to slot {}".format(
                    obsrv, slot_name))

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

    #########################
    #      Get & Clear      #
    #########################

    def get_slot(self, slot_name):
        return self.ontology.get_slot(slot_name)

    def get_intent(self, intent_name):
        return self.ontology.get_intent(intent_name)

    def clear_intent(self, intent_name):
        self.ontology.get_intent(intent_name).clear()

    def clear_slot(self, slot_name):
        self.ontology.get_slot(slot_name).clear()

    def to_json(self):
        """
        Serialize to json object 
        """
        obj = {
            "sysintent": self.sysintent.to_json(),
            "ontology": self.ontology.to_json(),
            "execution": self.execution
        }
        return obj

    def from_json(self, obj):
        """
        Load from serialized json
        """
        self.sysintent.from_json(obj['sysintent'])
        self.ontology.from_json(obj["ontology"])
        self.execution = obj["execution"]

    def intent_to_list(self, intent_name):
        """
        Get intent slot features
        """
        intent_slot = self.get_slot('intent')
        intent_node = self.get_intent(intent_name)
        intent_conf = intent_slot.value_conf_map.get(intent_name, 0.0)

        feat = []
        feat += [intent_conf]
        feat += intent_node.to_list()
        return feat

    def to_list(self):
        """
        State Feature 
        """
        feature = []
        ####################
        #       Slots      #
        ####################

        # Top Ks
        topks = []
        for node in self.ontology.slots.values():
            topks += node.to_list()
        feature += topks

        ######################
        #   Global features  #
        ######################
        # num_executions
        num_executions = 0

        h = []
        buckets = [0, 1, 2, 3, 5]

        h = [0.] * len(buckets)
        for idx, bucket_size in enumerate(buckets):
            if num_executions >= bucket_size:
                h[idx] = 1.0
                break
        feature += h

        # turn_id
        #turn_id = self.turn_id - 1
        # turn_id_max = 30  # Should set to same as user patience
        #t = [0.] * turn_id_max
        #t[turn_id] = 1.0
        #feature += t

        return feature
