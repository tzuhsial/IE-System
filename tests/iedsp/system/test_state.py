from iedsp.core import SysIntent
from iedsp.ontology import OntologyEngine
from iedsp.system import State
from iedsp import util


def test_state():

    ontology_file = "imageedit.ontology.json"
    ontology_json = util.load_from_json(ontology_file)
    visionengine = None

    ontology = OntologyEngine(visionengine, ontology_json)

    state = State(ontology)

    # Turn 1: this basically does nothing...
    turn_id = 1
    dialogue_act = {
        'slot': 'dialogue_act',
        'value': 'inform',
        'conf': 0.6
    }
    intent = {}
    slots = []
    state.update(dialogue_act, intent, slots, turn_id)
    assert state.pull() == SysIntent(request_slots=[{'slot': 'intent'}])

    # Turn id 2
    turn_id = 2
    dialogue_act = {
        'slot': 'dialogue_act',
        'value': 'inform',
        'conf': 0.8
    }
    intent = {
        'slot': 'intent',
        'value': 'open',
        'conf': 0.6
    }
    slots = []
    state.update(dialogue_act, intent, slots, turn_id)
    assert state.pull() == SysIntent(confirm_slots=[intent])

    # Turn id 3
    turn_id = 3
    dialogue_act = {
        'slot': 'dialogue_act',
        'value': 'affirm',
        'conf': 0.8
    }
    intent = {}
    slots = []
    state.update(dialogue_act, intent, slots, turn_id)
    assert state.pull() == SysIntent(request_slots=[{'slot': "image_path"}])

    # Turn id 4
    turn_id = 4
    dialogue_act = {
        'slot': 'dialogue_act',
        'value': 'inform',
        'conf': 0.8
    }
    intent = {}
    slots = [{'slot': 'image_path', 'value': 'some_value', 'conf': 1.0}]
    state.update(dialogue_act, intent, slots, turn_id)
    assert state.pull().executable()

    # Test frame stacking
    state.stack_intent("open")

    open_tree = state.get_intent("open")
    stacked_open_tree = state.framestack.top()

    assert open_tree.name == stacked_open_tree.name
    assert open_tree.intent == stacked_open_tree.intent

    open_tree.clear()

    assert open_tree.pull() != stacked_open_tree.pull()
    assert open_tree.children["image_path"].get_max_value() != \
        stacked_open_tree.children["image_path"].get_max_value()
