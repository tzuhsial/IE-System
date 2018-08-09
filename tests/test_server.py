import configparser

from iedsp.core import SystemAct
from iedsp.photoshop import PhotoshopPortal
from iedsp.system import System
from iedsp.util import build_slot_dict, imread, img_to_b64


def extract_das(system_act):
    das = []
    for sys_act in system_act['system_acts']:
        da = sys_act['dialogue_act']['value']
        das.append(da)
    return das


def test_system_demo():

    config = configparser.ConfigParser()
    config.read('config.dev.ini')

    image = imread('./images/3.jpg')
    b64_img_str = img_to_b64(image)

    system = System(config)
    photoshop = PhotoshopPortal(config["PHOTOSHOP"])
    system.reset()
    photoshop.reset()

    # Turn 1: open "./images/3.jpg"
    photoshop.observe({})
    ps_act = photoshop.act()

    user_act = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': build_slot_dict('intent', 'load', 1.0),
                'slots': [
                    build_slot_dict('b64_img_str', b64_img_str, 1.0)
                ]
            }
        ]
    }

    system.observe(ps_act)
    system.observe(user_act)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.EXECUTE, SystemAct.ASK]

    # Turn 2: select the dog

    photoshop.observe(system_act)
    ps_act = photoshop.act()

    user_act = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': build_slot_dict('intent', 'select_object', 1.0),
                'slots': []
            }
        ]
    }

    ps_act = photoshop.act()

    system.observe(ps_act)
    system.observe(user_act)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.REQUEST]

    # Turn 3: select the dog

    photoshop.observe(system_act)
    ps_act = photoshop.act()

    user_act = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': build_slot_dict('intent', 'select_object', 1.0),
                'slots': [build_slot_dict('object', 'dog', 0.7)]
            }
        ]
    }

    ps_act = photoshop.act()

    system.observe(ps_act)
    system.observe(user_act)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.REQUEST_LABEL]

    # Turn 4: object_mask_id 1
    photoshop.observe(system_act)
    ps_act = photoshop.act()

    user_act = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': build_slot_dict('intent', 'select_object', 1.0),
                'slots': [build_slot_dict('object_mask_id', '1', 1.0)]
            }
        ]
    }

    ps_act = photoshop.act()

    system.observe(ps_act)
    system.observe(user_act)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.EXECUTE, SystemAct.ASK]
    assert system.state.get_slot('object').get_max_value() == "dog"
    assert system.state.get_slot('object_mask_id').get_max_value() == "1"

    # Turn 5: adjust brightness
    photoshop.observe(system_act)
    ps_act = photoshop.act()

    user_act = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': build_slot_dict('intent', 'adjust', 1.0),
                'slots': [build_slot_dict('attribute', 'brightness', 0.5)]
            }
        ]
    }

    ps_act = photoshop.act()

    system.observe(ps_act)
    system.observe(user_act)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.CONFIRM]

    # Turn 5: Yes
    photoshop.observe(system_act)
    ps_act = photoshop.act()

    user_act = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', 'affirm', 0.9),
            }
        ]
    }

    ps_act = photoshop.act()

    system.observe(ps_act)
    system.observe(user_act)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.REQUEST]

    # Turn 6: +30
    photoshop.observe(system_act)
    ps_act = photoshop.act()

    user_act = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': build_slot_dict('intent', 'adjust', 1.0),
                'slots': [build_slot_dict('adjust_value', 30, 1.0)]
            }
        ]
    }

    ps_act = photoshop.act()

    system.observe(ps_act)
    system.observe(user_act)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.EXECUTE, SystemAct.ASK]
