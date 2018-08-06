from collections import defaultdict
import json
import logging

from .node import builder as nodelib
from .. import util

logger = logging.getLogger(__name__)


def OntologyPortal(config):
    if config["ONTOLOGY"] == "ImageEditOntology":
        return ImageEditOntology(config)
    else:
        logging.error("Unknown ontology: {}".format(config["ONTOLOGY"]))
        return None


class OntologyEngine(object):
    """
    Base Class for Ontology Definitions
    Attributes:
        visionengine (object):
        intents (dict)
        slots (dict)
    """

    def __init__(self, visionengine, ontology_config):

        # Load visionengine
        if visionengine is None:
            logger.warning(
                "Vision Engine is not initialized in OntologyEngine!")
        self.visionengine = visionengine

        # Build intents & slots
        ontology_json = util.load_from_json(ontology_config['ONTOLOGY_FILE'])
        self.intents, self.slots = self.build_from_json(ontology_json)

    def build_from_json(self, ontology_json):
        """
        Builds from ontology json file
        """
        # First create slots
        logger.debug("Building slots...")
        slots = {}
        for slot_json in ontology_json['slots']:

            name = slot_json["name"]
            threshold = slot_json["threshold"]
            possible_values = slot_json.get("possible_values", None)

            args = {
                "name": name,
                "threshold": threshold,
                "possible_values": possible_values,
            }

            use_visionengine = slot_json.get("use_visionengine", False)
            if use_visionengine:
                args['visionengine'] = self.visionengine

            node_name = slot_json["node"]

            slots[name] = nodelib(node_name)(**args)

        logger.debug("Building dependencies...")

        for slot_json in ontology_json["slots"]:
            name = slot_json["name"]
            children_list = slot_json["children"]
            parent_node = slots[name]
            for child_json in children_list:
                child_name = child_json["name"]
                optional = child_json.get("optional", False)

                child_node = slots[child_name]

                parent_node.add_child(child_node, optional)

        logger.debug("Building intents...")
        intents = {}
        for intent_json in ontology_json["intents"]:
            name = intent_json["name"]
            args = {
                "name": name
            }
            node_name = intent_json.get("node", "IntentNode")
            intents[name] = nodelib(node_name)(**args)

            for child_json in intent_json["children"]:
                child_name = child_json["name"]
                optional = child_json.get("optional", False)

                child_node = slots[child_name]

                intents[name].add_child(child_node, optional)

        self.intents = intents
        self.slots = slots

        logger.info("Done.")
        return self.intents, self.slots
