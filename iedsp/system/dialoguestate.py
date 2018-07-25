import copy
import json
import logging
import sys

from .core import UserAct

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
        obj = {}
        obj['slot'] = self.name
        max_value, max_conf = self.get_max_conf_value()
        obj['value'] = max_value
        obj['conf'] = max_conf
        return obj


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

    def __init__(self, name, confirm_threshold, default_values=[], permit_new_value=True, value_validator=None):
        """
        Args:
            name (str): slot name
            values (list): value names used as default
            permit_new_value (bool): whether an unseen value in the value list can be added
            value_validator (function): a function that validates the observed values
        """
        self.name = name
        self.confirm_threshold = confirm_threshold
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
        return 0. < max_conf < self.confirm_threshold

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

    def __init__(self, name, confirm_threshold, default_values=[], permit_new_value=True, value_validator=None):
        """
        Args:
            name (str): slot name
            values (list): unused
            value_validator (function): a function that validates the input value
        """
        assert len(default_values) <= 1
        super(PSToolSlot, self).__init__(
            name, confirm_threshold, default_values, permit_new_value, value_validator)

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

        # Slots
        self.request_slots = []
        self.confirm_slots = []
        self.query_slots = []
        self.execute_slots = []

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
            return False

    def pprint(self):
        """
        Pretty print out the current status of the table
        """
        print(json.dumps({'domain': self.name}))
        for slot in self.slots.values():
            print(json.dumps(slot.to_json()))

    def update_slots(self):
        """
        Categorize slots by whether value is present and confirm threshold
        Returns:
            confirm_slots (list): list of slot dicts
            request_slots (list): list of slot names
            query_slots (list): list of slot dicts
            execute_slots (list): list of slot dicts
        """
        self.confirm_slots = []
        self.request_slots = []
        self.query_slots = []
        self.execute_slots = []

        for slot in self.slots.values():
            slot_dict = slot.to_json()
            if slot.needs_request():
                self.request_slots.append(slot_dict)
            elif slot.needs_confirm():
                self.confirm_slots.append(slot_dict)
            elif slot.needs_query():
                self.query_slots.append(slot_dict)
            elif slot.executable():
                self.execute_slots.append(slot_dict)

        return self.confirm_slots, self.request_slots, self.query_slots, self.execute_slots

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

    def __init__(self, ontology, confirm_threshold):
        """
        In the constructor, we first build the slots and domain tables

        Args:
            ontology
        """

        # Records intent stack and history
        self.domainStack = list()

        # Build Domain Belief Slot that record
        self.domainSlot = BeliefSlot(
            "domain", confirm_threshold, default_values=ontology.domain_slot_map.keys())

        # Build Slots first
        slots = {}
        for slot_name, slot_class_name in ontology.slot_class_map.items():
            if slot_class_name == "BeliefSlot":
                slot = BeliefSlot(slot_name, ontology.slot_confirm_map.get(
                    slot_name, confirm_threshold))
            elif slot_class_name == "PSToolSlot":
                slot = PSToolSlot(slot_name, ontology.slot_confirm_map.get(
                    slot_name, confirm_threshold))
            else:
                raise ValueError(
                    "Unknown slot_class_name: {}".format(slot_class_name))
            slots[slot.name] = slot

        # Build Domain dependencies
        domains = {}
        for domain_name, slot_names in ontology.domain_slot_map.items():
            domain_slots = {slot_name: slots[slot_name]
                            for slot_name in slot_names}
            domains[domain_name] = DomainTable(domain_name, domain_slots)

        self.slots = slots
        self.domains = domains

    def update(self, dialogue_act, domain, slots, turn_id):
        """
        Update the dialogue state according to dialogue acts
        Args:
            dialogue_act
            domain
            slots
            turn_id
        """
        #
        if dialogue_act in [UserAct.AFFIRM, UserAct.NEGATE]:
            domainTable = self.getCurrentDomainTable()
            if len(domainTable.confirm_slots) == 0:
                logger.error("Confirm slots empty!")
                return False

            slot_dict = domainTable.confirm_slots.pop(0)
            new_observation = copy.deepcopy(slot_dict)
            new_observation['turn_id'] = turn_id
            if domain == UserAct.AFFIRM:
                new_observation['conf'] = 1.
                domainTable.add_new_observation(**new_observation)
            else:  # UserAct.NEGATE
                new_observation['conf'] = -1.  # set conf to -1.
                domainTable.add_new_observation(**new_observation)
            return True
        elif dialogue_act in [UserAct.INFORM]:
            pass

    def updateDomain(self, dialogue_act, domain, slots, turn_id):
        """
        Args:
            dialogue_act (str): user dialogue act
            domain (str): domain name
            slots (list): list of slot dict in { "slot": slot, "value": value, "conf": conf} format
        Returns:
            bool : True is succesfully updated else False
        """
        if domain in [UserAct.AFFIRM, UserAct.NEGATE]:
            domainTable = self.getCurrentDomainTable()
            if len(domainTable.confirm_slots) == 0:
                logger.error("Confirm slots empty!")
                return False

            slot_dict = domainTable.confirm_slots.pop(0)
            new_observation = copy.deepcopy(slot_dict)
            new_observation['turn_id'] = turn_id
            if domain == UserAct.AFFIRM:
                new_observation['conf'] = 1.
                domainTable.add_new_observation(**new_observation)
            else:  # UserAct.NEGATE
                new_observation['conf'] = -1.  # set conf to -1.
                domainTable.add_new_observation(**new_observation)
            return True

        if domain not in self.domains:
            logger.debug("Update failed, unknown domain: {}".format(domain))
            raise ValueError

        if domain in self.domainStack:
            # Remove from stack
            index = self.domainStack.index(domain)
            self.domainStack.pop(index)

        # Add to top of stack
        self.domainStack.insert(0, domain)

        domainTable = self.domains[domain]
        for slot_dict in slots:
            new_observation = copy.deepcopy(slot_dict)
            new_observation['turn_id'] = turn_id
            domainTable.add_new_observation(**new_observation)

        # Update request, confirm, query, execute slots
        domainTable.update_slots()
        return True

    def getCurrentDomainTable(self):
        """
        Gets current domain table, which is the top of the domainStack
        Returns:
            domainTable if domainStack has domain, else None
        """
        if len(self.domainStack) == 0:
            return None
        domain = self.domainStack[0]
        domainTable = self.domains[domain]
        return domainTable

    def clear(self, domain=None):
        """
        Clears a single domain table if it is specified
        """
        if domain is not None:
            self.domains[domain].clear()
        else:
            for domain in self.domains.values():
                domain.clear()


if __name__ == "__main__":
    pass
