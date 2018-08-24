from iedsp.ontology.node import *
from iedsp.util import build_slot_dict, sort_slots_with_key
from iedsp.util import imread, img_to_b64
from iedsp.visionengine import DummyClient


def test_beliefnode():

    node = BeliefNode("test")
    node.add_observation('a', 0.5, 1)
    assert node.get_max_conf_value() == ('a', 0.5)
    assert node.last_update_turn_id == 1

    node.add_observation('b', 0.3, 2)
    assert node.get_max_conf_value() == ('a', 0.4)
    assert node.last_update_turn_id == 2

    node.add_observation('c', 1.0, 3)
    assert node.get_max_conf_value() == ('c', 1.0)
    assert node.last_update_turn_id == 3

    assert node.to_json() == {'slot': "test", 'value': 'c', 'conf': 1.0}

    node.clear()
    assert node.get_max_conf_value() == (None, 0.0)
    assert node.last_update_turn_id == 0

    assert node.to_json() == {'slot': 'test'}

    # Test pull
    i1 = node.pull()
    i2 = node.pull()
    assert i1 == i2


def test_intentnode():
    open = IntentNode('open')

    assert open.get_max_conf_value() == ('open', 1.0)
    assert open.to_json() == build_slot_dict('intent', 'open', 1.0)


def test_pstoolnode():
    image_path = PSToolNode('image_path')

    assert image_path.add_observation('image_path1', 1.0, 1)
    assert image_path.get_max_conf_value() == ('image_path1', 1.0)
    assert image_path.last_update_turn_id == 1
    assert not image_path.add_observation('image_path', 0.8, 2)
    assert image_path.get_max_conf_value() == ('image_path1', 1.0)
    assert image_path.last_update_turn_id == 1
    assert image_path.add_observation('image_path2', 1.0, 2)
    assert image_path.get_max_conf_value() == ('image_path2', 1.0)
    assert image_path.last_update_turn_id == 2


def test_intent_pull():
    open = IntentNode('open')
    image_path = PSToolNode('image_path')

    # add_child
    assert open.add_child(image_path) == True
    assert open.add_child(image_path) == False

    # turn 0
    open_intent = open.pull()
    assert open_intent == SysIntent(
        request_slots=[build_slot_dict('image_path')])

    # turn 1
    turn_id = 1
    image_path.add_observation('ip1', 1.0, turn_id)
    open_intent = open.pull()

    assert open_intent == SysIntent(
        execute_slots=[build_slot_dict('image_path', 'ip1', 1.0)])

    assert image_path.last_update_turn_id == turn_id
    assert open.last_update_turn_id == -1

    # turn 2
    turn_id = 2
    image_path.add_observation('ip2', 1.0, turn_id)
    open_intent = open.pull()

    assert image_path.intent == SysIntent(
        execute_slots=[build_slot_dict('image_path', 'ip2', 1.0)])

    # turn 3
    turn_id = 3
    open_intent = open.pull()
    assert image_path.intent == SysIntent(
        execute_slots=[build_slot_dict('image_path', 'ip2', 1.0)])


def test_select_object_intent():
    # select_object_domain
    select = IntentNode('select_object')

    # object_mask_slot
    object_mask_slot = ObjectMaskStrNode('object_mask_str')

    b64_img_str_slot = PSToolNode('b64_img_str')
    object_slot = BeliefNode('object')
    object_mask_id_slot = BeliefNode('object_mask_id')

    position_slot = BeliefNode('position')
    adjective_slot = BeliefNode('adjective')

    # Define dependencies here
    # select_object
    # |- object_mask
    #    |- b64_img_str
    #    |- object
    #       |- position(optional)
    #       |- adjective(optional)
    #    |- object_mask_id

    select.add_child(object_mask_slot)
    object_mask_slot.add_child(b64_img_str_slot)
    object_mask_slot.add_child(object_slot)
    object_mask_slot.add_child(object_mask_id_slot)

    object_slot.add_child(position_slot, optional=True)
    object_slot.add_child(adjective_slot, optional=True)

    # Get b64_img_str first
    image_path = './images/3.jpg'
    image = imread(image_path)
    b64_img_str = img_to_b64(image)

    # Turn 0
    # User: I want to select
    # System: What object do you want?
    select_intent = select.pull()
    assert select_intent == SysIntent(
        request_slots=[build_slot_dict('object')])

    # Turn 1
    # User: select the right dog: inform(intent=select, object=dog, position=right)
    # System: Let me confirm.  You want position=right?: confirm(position=right)
    turn_id = 1

    b64_img_str_slot.add_observation(b64_img_str, 1.0, turn_id)
    object_slot.add_observation('dog', 1.0, turn_id)
    position_slot.add_observation('right', 0.6, turn_id)

    select_intent = select.pull()
    assert select_intent == SysIntent(confirm_slots=[build_slot_dict('position', 'right', 0.6)],
                                      execute_slots=[build_slot_dict('object', 'dog', 1.0)])

    # Turn 2:
    # User: Yes: confirm(position=right)
    # System: I cannot find what you want. Can you label it for me?: request
    turn_id = 2
    position_slot.add_observation('right', 0.9, turn_id)

    select_intent = select.pull()
    assert select_intent == SysIntent(
        query_slots=[build_slot_dict('object', 'dog', 1.0),
                     build_slot_dict('position', 'right', 0.9)]
    )

    # Turn 3:
    # User: (Selects with Photoshop)
    # System: Executes
    turn_id = 3
    object_mask_slot.add_observation("test_mask_str", 1.0, turn_id)
    select_intent = select.pull()
    assert select_intent == SysIntent(
        execute_slots=[build_slot_dict(
            'object_mask_str', 'test_mask_str', 1.0)]
    )
