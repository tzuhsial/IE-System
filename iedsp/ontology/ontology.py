import json
import logging

from .node import builder as nodelib
from .validator import builder as vallib
from ..visionengine import VisionEnginePortal
from .. import util

logger = logging.getLogger(__name__)


def OntologyPortal(global_config):
    visionengine_config = global_config["VISIONENGINE"]
    visionengine = VisionEnginePortal(visionengine_config)

    ontology_file = global_config["DEFAULT"]["ONTOLOGY_FILE"]
    ontology_json = util.load_from_json(ontology_file)

    ontology = OntologyEngine(visionengine, ontology_json)
    return ontology


class OntologyEngine(object):
    """
    Ontology Engine
    Builds intent & slot dependency graphs (for user intents)
    There should also be a special intent tree used to classify current intent 
    intent
    |- intent
    Attributes:
        visionengine (object):
        intents (dict)
        slots (dict)
    """

    INTENT = "intent"

    def __init__(self, visionengine, ontology_json):

        # Load visionengine
        if visionengine is None:
            logger.warning(
                "Vision Engine is not initialized in OntologyEngine!")
        self.visionengine = visionengine

        self.intents = {}
        self.slots = {}

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
            val_name = slot_json.get("validator", "")
            val_cls = vallib(val_name)
            if val_cls is not None:
                validator = val_cls()
            else:
                validator = None

            args = {
                "name": name,
                "threshold": threshold,
                "possible_values": possible_values,
                "validator": validator,
            }

            use_visionengine = slot_json.get("use_visionengine", False)
            if use_visionengine:
                args['visionengine'] = self.visionengine

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
