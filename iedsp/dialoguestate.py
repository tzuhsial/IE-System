import copy
import json
import logging
import sys

from .core import UserAct, Hermes
from .ontology import getOntologyWithName

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Slot(object):
    """
        Base class for slots
    """

    def __init__(self):
        raise NotImplementedError

    def add_new_observation(self):
        raise NotImplementedError

    def get_max_conf_value(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def to_json(self):
        """ 
        Returns:
            obj (dict): slot dict
        """
        max_value, max_conf = self.get_max_conf_value()
        if max_value is None:
            return ActionHelper.build_slot_dict(self.name)
        else:
            return ActionHelper.build_slot_dict(self.name, max_value, max_conf)


class BeliefSlot(Slot):
    """
        A slot with probability distribution over a finite state of values.
        Used for Speech Input

        Example:
        ```
        slot = BeliefSlot("adjustValue", ["more", "less"], False)
        ```
        Attrbutes:
            name (str): name of the slot
            default_values (list): default values if provided
            value_conf_map (dict): entity_value -> conf mapping
            last_update_turn_id (int): the last turn this slot was modified
            permit_new_value (bool): whether this BeliefSlot permits new slots
            value_validator (function): a function to validate the values
    """

    def __init__(self, name, threshold, default_values=[], permit_new_value=True, value_validator=None):
        """
        Args:
            name (str): slot name
            values (list): value names used as default
            permit_new_value (bool): whether an unseen value in the value list can be added
            value_validator (function): a function that validates the observed values
        """
        self.name = name
        self.threshold = threshold
        self.default_values = default_values
        self.value_conf_map = {v: 0. for v in self.default_values}
        self.permit_new_value = permit_new_value
        self.last_update_turn_id = 0
        self.value_validator = value_validator

    def add_new_observation(self, value, conf, turn_id):
        """
        Args:
            value (str):
            conf  (str):
            turn_id (int):
        Returns:
            bool: True is successful, False otherwise
        """
        if self.value_validator is not None and not self.value_validator(value):
            return False
        if value not in self.value_conf_map and not self.permit_new_value:
            return False

        # Simply assign confidence for now
        self.value_conf_map[value] = conf
        self.last_update_turn_id = turn_id
        return True

    def get_max_conf_value(self):
        """
        Returns:
            value with the maximum confidence or None if value_conf_map is empty
        """
        if len(self.value_conf_map) == 0:
            return None, -1
        max_value, max_conf = max(
            self.value_conf_map.items(), key=lambda tup: tup[1])
        if max_conf < 0:  # For neglected stuff
            return None, -1
        return max_value, max_conf

    def to_json(self):
        """
        Returns:
            obj (dict): slot dict
        """
        obj = {}
        obj['slot'] = self.name
        max_value, max_conf = self.get_max_conf_value()
        if max_value is not None:
            obj['value'] = max_value
            obj['conf'] = max_conf
        return obj

    def needs_request(self):
        max_value, max_conf = self.get_max_conf_value()
        return max_value == None

    def needs_confirm(self):
        max_value, max_conf = self.get_max_conf_value()
        return 0. < max_conf < self.threshold

    def needs_query(self):
        return False

    def executable(self):
        return not self.needs_request() and not self.needs_confirm() and not self.needs_query()

    def clear(self):
        """
        Clears the value of this slot
        """
        self.value_conf_map = {v: 0 for v in self.default_values}
        self.last_update_turn_id = 0


class PSToolSlot(BeliefSlot):
    """
    Used for Photoshop Tool input
    The value of slot that is obtained by interacting directly with Photoshop.
    Therefore, only one value can be present, and has confidence 1.0
    """

    def __init__(self, name, threshold, default_values=[], permit_new_value=True, value_validator=None):
        """
        Args:
            name (str): slot name
            values (list): unused
            value_validator (function): a function that validates the input value
        """
        assert len(default_values) <= 1
        super(PSToolSlot, self).__init__(
            name, threshold, default_values, permit_new_value, value_validator)

    def add_new_observation(self, value, conf, turn_id):
        """
        Args:
            value (any): value for the slot
            turn_id (int): update turn id
        """
        self.value_conf_map = {v: 0 for v in self.default_values}
        result = super(PSToolSlot, self).add_new_observation(
            value, conf, turn_id)
        return result


class DomainTable(object):
    """
    Domain Table with corresponding slots 
    ------------------------------
    |           Domain           |
    ------------------------------
    |   slot  |  value  |  conf  |
    ------------------------------
    |   slot  |  value  |  conf  |
    ------------------------------
    """

    def __init__(self, name, slots):
        """
        Build domain tables
        Args:
            name (str): domain name
            slots (dict): slot -> slot_class mapping
        """
        self.name = name
        self.slots = slots

    def add_new_observation(self, slot, value, conf, turn_id):
        """
        Adds new observation to slot
        Returns:
            bool : True if successfully added else False
        """
        if slot in self.slots:
            self.slots[slot].add_new_observation(value, conf, turn_id)
            return True
        else:
            raise ValueError("Unknown slot: {}".format(slot))

    def pprint(self):
        """
        Pretty print out the current status of the table
        """
        print(json.dumps({'domain': self.name}))
        for slot in self.slots.values():
            print(json.dumps(slot.to_json()))

    def get_goal_slots(self):
        """
        Categorize slots by whether value is present and confirm threshold
        Returns:
            confirm_slots (list): list of slot dicts
            request_slots (list): list of slot names
            query_slots (list): list of slot dicts
            execute_slots (list): list of slot dicts
        """
        request_slots = []
        confirm_slots = []
        query_slots = []
        execute_slots = []

        for slot in self.slots.values():
            slot_dict = slot.to_json()
            if slot.needs_request():
                request_slots.append(slot_dict)
            elif slot.needs_confirm():
                confirm_slots.append(slot_dict)
            elif slot.needs_query():
                query_slots.append(slot_dict)
            elif slot.executable():
                execute_slots.append(slot_dict)

        return request_slots, confirm_slots, query_slots, execute_slots

    def executable(self):
        """
        Determine whether this table is ready to be executed
        Returns:
            bool : whether this domain is executable
        """
        _, _, _, execute_slots = self.get_goal_slots()
        return len(execute_slots) == len(self.slots)

    def clear(self):
        """
        Clears slot values of this table
        """
        for slot in self.slots.values():
            slot.clear()


class DialogueState(object):
    """
    Multi-domain dialogue state based on Ontology
    Main purpose is to record slot values
    Attributes:
        slots (dict): slot -> slot_class mapping
        domains (dict): domain -> domain_class mapping
        domainStack (list): list of domain
    """

    def __init__(self, config):
        """
        In the constructor, we first build the slots and domain tables

        Args:
            ontology
        """
        # Get ontology
        self.ontology = getOntologyWithName(config["ONTOLOGY"])
        self.default_thresh = float(config["DEFAULT_THRESHOLD"])

        # Build Slots first
        slots = {}
        for slot_name, ont_slot_class in self.ontology.slots.items():
            slot_thresh_key = slot_name.upper() + "_THRESHOLD"
            slot_thresh = config.get(slot_thresh_key, self.default_thresh)

            # Perhaps write a classmethod in the future
            construct_args = copy.deepcopy(vars(ont_slot_class))
            construct_args.pop('slot_type')
            construct_args['threshold'] = slot_thresh

            if ont_slot_class.slot_type == "BeliefSlot":
                slot = BeliefSlot(**construct_args)
            elif ont_slot_class.slot_type == "PSToolSlot":
                slot = PSToolSlot(**construct_args)
            else:
                raise ValueError(
                    "Unknown slot_type: {}".format(ont_slot_class.slot_type))
            slots[slot.name] = slot

        # Add a special dialogue_act table
        dialogue_act_thresh = config.get(
            "DIALOGUE_ACT_THRESHOLD", self.default_thresh)
        dialogue_act_slot = BeliefSlot(
            self.ontology.DIALOGUE_ACT, dialogue_act_thresh)
        slots[dialogue_act_slot.name] = dialogue_act_slot

        # Build Domain dependencies
        tables = {}
        for domain_name, ont_domain_class in self.ontology.domains.items():
            domain_slot_names = ont_domain_class.get_slot_names()
            domain_slots = {slot_name: slots[slot_name]
                            for slot_name in domain_slot_names}
            tables[domain_name] = DomainTable(domain_name, domain_slots)

        # Add a special dialogue_act table
        dialogue_act_slots = {dialogue_act_slot.name: dialogue_act_slot}
        dialogue_act_table = DomainTable(
            self.ontology.DIALOGUE_ACT, dialogue_act_slots)
        tables[dialogue_act_table.name] = dialogue_act_table

        # Add an dialogue_act table to determine dialogue_act
        self.dialogue_act_table = dialogue_act_table
        self.domain_stack = list()

        self.slots = slots
        self.tables = tables

        # Request, Confirm, Query, Execute are defined according to the domain
        self.request_slots = []
        self.confirm_slots = []
        self.query_slots = []
        self.execute_slots = []

    def update(self, dialogue_act, slots, turn_id):
        """
        Update the dialogue state according to dialogue acts
        Args:
            dialogue_act (dict): a slot dict 
            slots (list): list of slot dicts
            turn_id (int): turn index
        """
        # TODO: dialogue_act & confirm confidence
        user_dialogue_act = dialogue_act['value']

        # Current domain
        if user_dialogue_act in UserAct.domain_acts():
            self.prioritize_domain(user_dialogue_act)
            self.update_domain(user_dialogue_act, slots, turn_id)
        elif user_dialogue_act in UserAct.confirm_acts():
            domain_table = self.get_current_domain()
            confirm_dict = self.confirm_slots.pop(0)
            new_observation = copy.deepcopy(confirm_dict)
            new_observation['turn_id'] = turn_id
            new_observation['conf'] = 1. if user_dialogue_act == UserAct.AFFIRM else -1.
            domain_table.add_new_observation(**new_observation)
        self.load_domain_goal_slots()

    def update_dialogue_act(self, dialogue_act, turn_id):
        """
        Update dialogue_act tables
        Returns:
            bool: whether we are sure of this dialogue_act
        """
        # Update Domain Classifier Table first
        dialogue_act_observation = copy.deepcopy(dialogue_act)
        dialogue_act_observation['slot'] = self.ontology.DIALOGUE_ACT
        dialogue_act_observation['turn_id'] = turn_id
        self.dialogue_act_table.add_new_observation(**dialogue_act_observation)

        domain_name = dialogue_act['value']
        domain_table = self.tables[domain_name]

        return self.dialogue_act_table.executable()

    def update_domain(self, domain_name, slots, turn_id):
        """
        Args:
            dialogue_act (dict): {'slot': 'dialogue_act', 'value': v1, 'conf': c}
            slots (list): list of slot dict in { "slot": slot, "value": value, "conf": conf} format
            turn_id (int): turn index
        Returns:
            bool : True is succesfully updated else False
        """
        domain_table = self.tables[domain_name]
        for slot_dict in slots:
            new_observation = copy.deepcopy(slot_dict)
            new_observation['turn_id'] = turn_id
            domain_table.add_new_observation(**new_observation)

    def prioritize_domain(self, domain_name):
        """
        Move domain_table to top of stack
        """
        # Move domain_table to the top of stack
        domain_table = self.tables[domain_name]
        if domain_table in self.domain_stack:
            index = self.domain_stack.index(domain_table)
            self.domain_stack.pop(index)
        self.domain_stack.insert(0, domain_table)

    def load_domain_goal_slots(self, domain_name=None):
        """
        Get goal slots of the specific domain
        """
        if domain_name is None:
            domain_table = self.domain_stack[0]
        else:
            domain_table = self.tables[domain_name]

        self.request_slots, self.confirm_slots, self.query_slots, self.execute_slots = \
            domain_table.get_goal_slots()

    def print_goals(self):
        print("Dialogue State:")
        print("request_slots", self.request_slots)
        print("confirm_slots", self.confirm_slots)
        print("query_slots", self.query_slots)
        print("execute_slots", self.execute_slots)

    def get_current_domain(self):
        """
        Gets current domain table, which is the top of the domainStack
        Returns:
            domain_table if domainStack has domain, else None
        """
        if len(self.domain_stack) == 0:
            return None
        return self.domain_stack[0]

    def clear(self, domain=None):
        """
        Clears a single domain table if it is specified
        """
        if domain is not None:
            self.tables[domain].clear()
        else:
            for table in self.tables.values():
                table.clear()


if __name__ == "__main__":
    pass
