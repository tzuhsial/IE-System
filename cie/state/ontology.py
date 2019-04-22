from collections import OrderedDict
import json
import logging

from .node import builder as nodelib
from .validator import builder as vallib
from ..visionengine import VisionEnginePortal
from .. import util

logger = logging.getLogger(__name__)


def OntologyPortal(global_config):
    ontology_file = global_config["DEFAULT"]["ONTOLOGY_FILE"]
    ontology_json = util.load_from_json(ontology_file)

    ontology = OntologyEngine(ontology_json)
    return ontology


class OntologyEngine(object):
    """
    Ontology Engine
    Builds intent & slot dependency graphs (for user intents)
    There should also be a special intent tree used to classify current intent 
    intent
    |- intent
    Attributes:
        intents (dict)
        slots (dict)
    """

    INTENT = "intent"

    def __init__(self, ontology_json):

        self.intents = OrderedDict()
        self.slots = OrderedDict()

        self.build_from_json(ontology_json)

        assert 'intent' in self.intents and 'intent' in self.slots, "Ontology Engine requires an intent tree to classify intents!"

    def build_from_json(self, ontology_json):
        """
        Builds from ontology json file
        """
        # First create slots
        logger.debug("Building slots...")
        for slot_json in ontology_json['slots']:

            name = slot_json["name"]
            threshold = slot_json["threshold"]
            possible_values = slot_json.get("possible_values", None)

            # Instantiate object
            validator = None
            if slot_json.get("validator") is not None:
                val_name = slot_json.get("validator")
                validator = vallib(val_name)()

            args = {
                "name": name,
                "threshold": threshold,
                "possible_values": possible_values,
                "validator": validator,
            }

            node_name = slot_json["node"]

            self.slots[name] = nodelib(node_name)(**args)

        logger.debug("Building dependencies...")

        for slot_json in ontology_json["slots"]:
            name = slot_json["name"]
            children_list = slot_json["children"]
            parent_node = self.slots[name]
            for child_json in children_list:
                child_name = child_json["name"]
                optional = child_json.get("optional", False)
                child_node = self.slots[child_name]

                parent_node.add_child(child_node, optional)

        logger.debug("Building intents...")
        for intent_json in ontology_json["intents"]:
            name = intent_json["name"]
            args = {
                "name": name
            }
            node_name = intent_json.get("node", "IntentNode")
            self.intents[name] = nodelib(node_name)(**args)

            for child_json in intent_json["children"]:
                child_name = child_json["name"]
                optional = child_json.get("optional", False)
                child_node = self.slots[child_name]
                self.intents[name].add_child(child_node, optional)

            # intent nodes store their children and descendants
            self.intents[name].build_node_dict()

        logger.info("Done.")
        return self.intents, self.slots

    def get_intent(self, name):
        if name not in self.intents:
            logger.error("Unknown intent: {}".format(name))
            return None
        return self.intents[name]

    def get_slot(self, name):
        if name not in self.slots:
            logger.error("Unknown slot: {}".format(name))
            return None
        return self.slots[name]

    def clear(self):
        for intent_tree in self.intents.values():
            intent_tree.clear()

    def flush(self):
        for intent_tree in self.intents.values():
            intent_tree.flush()

    def to_json(self):
        """
        """
        obj = {}
        for slot_name, slot_node in self.slots.items():
            obj[slot_name] = {
                # consistency
                'value_conf': sorted(slot_node.value_conf_map.items()),
                'last_update_turn_id': slot_node.last_update_turn_id
            }
        return obj

    def from_json(self, obj):
        """
        Load slot values from obj 
        """
        for slot_name, slot_obj in obj.items():
            self.slots[slot_name].value_conf_map = dict(slot_obj["value_conf"])
            self.slots[slot_name].last_update_turn_id = slot_obj["last_update_turn_id"]
