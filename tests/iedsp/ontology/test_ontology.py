
from iedsp.ontology import OntologyEngine
from iedsp import util


def test_ontology():
    """
    Build ontology from slot dependencies
    """

    ontology_file = "imageedit.ontology.json"
    ontology_json = util.load_from_json(ontology_file)

    engine = OntologyEngine(ontology_json)

    for intent_json in ontology_json["intents"]:
        assert intent_json["name"] in engine.intents

    for slot_json in ontology_json["slots"]:
        assert slot_json["name"] in engine.slots

    # TODO: validate engine dependency graph, or validate ontology_file
    assert engine.slots['image_path'].add_observation(123, 1.0, 1) == False