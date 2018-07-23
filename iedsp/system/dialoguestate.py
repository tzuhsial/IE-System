import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Slot(object):
    """
        Base class for slots
    """

    def __init__(self):
        raise NotImplementedError

    def addNewObservation(self):
        raise NotImplementedError

    def getMaxConfValue(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def toString(self):
        raise NotImplementedError


class BeliefSlot(Slot):
    """
        A slot with probability distribution over a finite state of values.
        Used for Speech Input

        Example:
        ```
        slot = BeliefSlot("adjustValue", ["more", "less"], False)
        ```
        Attrbutes:
            value_conf_map (dict): entity_value -> conf mapping
            last_update_turn_id (int): the last turn this slot was modified
            permit_new_value (bool): whether this BeliefSlot permits new slots
    """

    def __init__(self, name, values=[], permit_new_value=True, value_validator=None):
        """
        Args:
            name (str): slot name
            values (list): value names used as default
            permit_new_value (bool): whether an unseen value in the value list can be added
            value_validator (function): a function that validates the observed values
        """
        self.name = name
        self.default_values = values
        self.value_conf_map = {v: 0. for v in self.default_values}
        self.permit_new_value = permit_new_value
        self.last_update_turn_id = 0
        self.value_validator = value_validator

    def addNewObservation(self, value, conf, turn_id):
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

    def getMaxConfValue(self):
        """
        Returns:
            value with the maximum confidence or None if value_conf_map is empty
        """
        if len(self.value_conf_map) == 0:
            return None, 0.
        max_value, max_conf = max(
            self.value_conf_map.items(), key=lambda tup: tup[1])
        return max_value, max_conf

    def toString(self):
        """
        Returns max conf value and cocatenate as string with slot name
        """
        ret = ""
        ret += "Slot: " + self.name
        max_value, max_conf = self.getMaxConfValue()
        ret += ", Value:" + str(max_value)
        ret += ", Conf:" + str(max_conf)
        return ret

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

    def __init__(self, name, values=[], value_validator=None):
        """
        Args:
            name (str): slot name
            values (list): unused
            value_validator (function): a function that validates the input value
        """
        self.name = name
        self.value = None
        self.value_validator = value_validator
        self.last_update_turn_id = 0

    def addNewObservation(self, value, conf, turn_id):
        """
        Args:
            value (any): value for the slot
            turn_id (int): update turn id
        """
        if self.value_validator is not None and not self.value_validator(value):
            return False
        elif conf != 1.:
            return False
        else:
            self.value = value
            self.last_update_turn_id = turn_id
        return True

    def getMaxConfValue(self):
        """
        """
        return self.value, 1.0

    def toString(self):
        """
        Returns max conf value and cocatenate as string with slot name
        """
        ret = ""
        ret += "Slot: " + self.name
        max_value, max_conf = self.getMaxConfValue()
        ret += ", Value:" + str(max_value)
        ret += ", Conf:" + str(max_conf)
        return ret

    def clear(self):
        self.value = None
        self.last_update_turn_id = 0


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

    def addNewObservation(self, slot, value, conf, turn_id):
        """
        Adds new observation to slot
        Returns:
            bool : True if successfully added else False
        """
        if slot in self.slots:
            self.slots[slot].addNewObservation(value, conf, turn_id)
            return True
        else:
            return False

    def pprint(self):
        """
        Pretty print out the current status of the table
        """
        print("Domain:", self.name)
        for slot in self.slots.values():
            print(slot.toString())

    def getCurrentSlots(self, confirm_threshold):
        """
        Divide slot values in table by whether value is present and confirm threshold
        Returns:
            confirm_slots (list): list of slot dicts
            request_slots (list): list of slot names
            query_slots (list): list of slot dicts
        """
        confirm_slots = []
        request_slots = []
        query_slots = []
        execute_slots = []

        for slot in self.slots.values():
            max_value, max_conf = slot.getMaxConfValue()
            if max_value is None:
                # Add to requests
                request_slots.append(slot.name)
            else:
                slot_dict = {'slot': slot.name,
                             'value': max_value, 'conf': max_conf}
                if max_conf < confirm_threshold:
                    confirm_slots.append(slot_dict)
                else:
                    query_slots.append(slot_dict)

        return confirm_slots, request_slots, query_slots

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

    def __init__(self, ontology):
        """
        Args:
            ontology
        """
        # Build Slots first
        slots = {}
        for slot_name, slot_class_name in ontology.slot_class_map.items():
            if slot_class_name == "BeliefSlot":
                slot = BeliefSlot(slot_name)
            elif slot_class_name == "PSToolSlot":
                slot = PSToolSlot(slot_name)
            else:
                raise ValueError(
                    "Unknown slot_class_name: {}".format(slot_class_name))
            slots[slot.name] = slot

        # Build Domain
        domains = {}
        for domain_name, slot_names in ontology.domain_slot_map.items():
            domain_slots = {slot_name: slots[slot_name]
                            for slot_name in slot_names}
            domains[domain_name] = DomainTable(domain_name, domain_slots)

        self.slots = slots
        self.domains = domains

        # Current system slots
        self.request_slots = list()
        self.confirm_slots = list()
        self.query_slots = list()
        self.execute_slots = list()
        self.mask_strs = dict()

        # Records intent stack and history
        self.domainStack = list()

    def update(self, domain, slots, turn_id):
        """
        Update domain slots
        Args:
            domain (str): domain name
            slots (list): list of slot dict in { "slot": slot, "value": value, "conf": conf} format
        Returns:
            bool : True is succesfully updated else False
        """
        if domain not in self.domains:
            logger.debug("Update failed, unknown domain: {}".format(domain))
            return False

        if domain in self.domainStack:
            # Remove from stack
            index = self.domainStack.index(domain)
            self.domainStack.pop(index)

        # Add to top of stack
        self.domainStack.insert(0, domain)

        domainTable = self.domains[domain]
        for slot_dict in slots:
            domainTable.addNewObservation(
                slot_dict['slot'], slot_dict['value'], slot_dict['conf'], turn_id)
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
