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

    def add_observation(self):
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
            return Hermes.build_slot_dict(self.name)
        else:
            return Hermes.build_slot_dict(self.name, max_value, max_conf)


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

    def add_observation(self, value, conf, turn_id):
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
            value (str): 
            conf (float):
        """
        if len(self.value_conf_map) == 0:
            return None, -1
        max_value, max_conf = max(
            self.value_conf_map.items(), key=lambda tup: tup[1])
        if max_conf <= 0:  # For neglected stuff
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
        """
        Whether the slot needs to be requested
        Returns:
            bool
        """
        max_value, max_conf = self.get_max_conf_value()
        return max_value == None

    def needs_confirm(self):
        """
        Whether the slot needs to be confirmed
        Returns:
            bool
        """
        max_value, max_conf = self.get_max_conf_value()
        return 0. < max_conf < self.threshold

    def needs_query(self):
        """
        Whether the slot needs to be queried
        Returns:
            bool
        """
        return False

    def executable(self):
        """
        Whether the slot can be executed
        Returns:
            bool
        """
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

    def add_observation(self, value, conf, turn_id):
        """
        Args:
            value (any): value for the slot
            turn_id (int): update turn id
        """
        self.value_conf_map = {v: 0 for v in self.default_values}
        result = super(PSToolSlot, self).add_observation(
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

    def add_observation(self, slot, value, conf, turn_id):
        """
        Adds new observation to slot
        Returns:
            bool : True if successfully added else False
        """
        if slot in self.slots:
            self.slots[slot].add_observation(value, conf, turn_id)
            return True
        else:
            raise ValueError("Unknown slot: {}".format(slot))

    def to_json(self):
        """
        Jsonify domain table
        """
        obj = {}
        obj['domain'] = self.name
        slots = []
        for slot in self.slots.values():
            slots.append(slot.to_json())
        obj['slots'] = slots
        return obj

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


class DomainStack(object):
    """
    A C++ like stack to store domain tables
    With some convenient functions e.g. prioritize
    """

    def __init__(self):
        self._stack = list()

    def __contains__(self, domain_table):
        return domain_table in self._stack

    def prioritize(self, domain_table):
        """
        Move domain_table to top of stack, 
        Regardless of whether is in stack already
        """
        if domain_table in self._stack:
            self._stack.remove(domain_table)
        self.push(domain_table)

    def top(self):
        if len(self._stack):
            return self._stack[0]
        return None

    def push(self, domain):
        assert isinstance(domain, DomainTable)
        self._stack.insert(0, domain)

    def pop(self):
        self._stack.pop(0)

    def size(self):
        return len(self._stack)


class DialogueState(object):
    """
    Multi-domain dialogue state based on Ontology
    Attributes:
        slots (dict): slot -> slot_class mapping
        domains (dict): domain -> domain_class mapping
        domain_stack (list): list of domain
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
        da_thresh = config.get(
            "DIALOGUE_ACT_THRESHOLD", self.default_thresh)
        da_slot = BeliefSlot(
            self.ontology.DIALOGUE_ACT, da_thresh)
        slots[da_slot.name] = da_slot

        # Build Domain dependencies
        tables = {}
        for domain_name, ont_domain_class in self.ontology.domains.items():
            domain_slot_names = ont_domain_class.get_slot_names()
            domain_slots = {slot_name: slots[slot_name]
                            for slot_name in domain_slot_names}
            tables[domain_name] = DomainTable(domain_name, domain_slots)

        # Add a special dialogue_act table
        dialogue_act_slots = {da_slot.name: da_slot}
        da_table = DomainTable(
            self.ontology.DIALOGUE_ACT, dialogue_act_slots)
        tables[da_table.name] = da_table

        # Add an dialogue_act table to determine dialogue_act
        self.da_slot = da_slot
        self.da_table = da_table
        self.domain_stack = DomainStack()

        self.slots = slots
        self.tables = tables

        # Goals: Request, Confirm, Query, Execute Slots
        self.request_slots = []
        self.confirm_slots = []
        self.query_slots = []
        self.execute_slots = []

    def update(self, dialogue_act, slots, turn_id):
        """
        Update the dialogue state according to dialogue acts
        Args:
            dialogue_act (dict): a slot dict for dialogue_act 
            slots (list): list of slot dicts
            turn_id (int): turn index
        """
        user_da = dialogue_act['value']
        if user_da in UserAct.domain_acts():
            self.update_domain_acts(dialogue_act, slots, turn_id)
        elif user_da in UserAct.confirm_acts():
            self.update_confirm_acts(dialogue_act, slots, turn_id)
        # Load goals from top of domain stack
        self.load_domain_goal_slots()

    def update_domain_acts(self, dialogue_act, slots, turn_id):
        """
        Update actions that are domain related
        """
        # Check current domain action
        da_obsrv = copy.deepcopy(dialogue_act)
        da_obsrv['turn_id'] = turn_id
        self.da_table.add_observation(**da_obsrv)

        if self.da_table.executable():
            # Get directly from da_slot
            domain_name, _ = self.da_slot.get_max_conf_value()
            # Move this table to top of domain_stack
            domain_table = self.tables[domain_name]
            self.domain_stack.prioritize(domain_table)
            # Clear dialogue act slot
            self.da_table.clear()
        elif self.domain_stack.top() != self.da_table:
            self.domain_stack.push(self.da_table)

        # Update slots
        for slot_dict in slots:
            slot_obsv = copy.deepcopy(slot_dict)
            slot_obsv['turn_id'] = turn_id
            slot_name = slot_obsv.pop('slot')
            self.slots[slot_name].add_observation(**slot_obsv)

    def update_confirm_acts(self, dialogue_act, slots, turn_id):
        """
        Update acts that are confirm related
        We don't seem to need slots
        TODO: confidence of AFFIRM, NEGATE
        """
        # Look at the domain which the user is confirming to
        domain_table = self.domain_stack.top()

        confirm_dict = self.confirm_slots.pop(0)
        confirm_obsv = copy.deepcopy(confirm_dict)
        confirm_obsv['turn_id'] = turn_id

        if dialogue_act['value'] == UserAct.AFFIRM:
            confirm_obsv['conf'] = 1.
        else:
            confirm_obsv['conf'] = -1.

        domain_table.add_observation(**confirm_obsv)

        # if top is da_table
        if domain_table.name == self.ontology.DIALOGUE_ACT:
            self.domain_stack.pop()
            if domain_table.executable():
                # Get directly from da_slot
                domain_name, _ = self.da_slot.get_max_conf_value()
                # Push to domain_stack if is new dialogue_act
                if self.domain_stack.size() == 0 or \
                        domain_name != self.domain_stack.top().name:
                    self.domain_stack.push(self.tables[domain_name])
                # Clear dialogue act slot
                self.da_table.clear()
            else:
                self.domain_stack.push(self.da_table)

    def load_domain_goal_slots(self, domain_name=None):
        """
        Get goal slots of the specific domain
        """
        if domain_name is None:
            domain_table = self.domain_stack.top()
        else:
            domain_table = self.tables[domain_name]

        self.request_slots, self.confirm_slots, self.query_slots, self.execute_slots = \
            domain_table.get_goal_slots()

    def print_stack(self):
        print("-------------------")
        print("|   Domain Stack  |")
        print("-------------------")
        for idx, table in enumerate(self.domain_stack._stack):
            print('| {:<15} |'.format(str(idx) + " " + table.name))
        print("-------------------")

    def print_goals(self):
        print("[System Goals]")
        print("[Request]", self.request_slots)
        print("[Confirm]", self.confirm_slots)
        print("[Query]", self.query_slots)
        print("[Execute]", self.execute_slots)

    def get_current_domain(self):
        """
        Gets the top of the domain_stack
        Returns:
            domain_table if domainStack has domain, else None
        """
        if self.domain_stack.size() == 0:
            return None
        else:
            return self.domain_stack.top()

    def clear_goals(self):
        self.request_slots = []
        self.confirm_slots = []
        self.query_slots = []
        self.execute_slots = []

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
