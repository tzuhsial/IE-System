import configparser

from iedsp.system import System
from iedsp.core import SysIntent, SystemAct
from iedsp import util


def extract_das(system_act):
    das = []
    for sys_act in system_act['system_acts']:
        da = sys_act['dialogue_act']['value']
        das.append(da)
    return das


def test_system():
    config = configparser.ConfigParser()
    config.read("config.dev.ini")

    system = System(config)
    system.reset()

    image_path = "./images/3.jpg"
    image = util.imread(image_path)
    b64_img_str = util.img_to_b64(image)

    # Turn 1: open image_path ./images/3.jpg
    observation = {
        'user_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': util.build_slot_dict('intent', 'open', 1.),
                'slots': [
                    util.build_slot_dict('image_path', image_path, 1.0)
                ]
            }
        ]
    }
    system.observe(observation)
    system_act = system.act()

    assert extract_das(system_act) == [
        SystemAct.EXECUTE, SystemAct.GREETING, SystemAct.ASK]

    # Turn id 2: select the dog
    observation = {
        'user_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': util.build_slot_dict('intent', 'select_object', 0.6),
                'slots': [
                    util.build_slot_dict('object', 'dog', 1.0)
                ]
            }
        ],
        'photoshop_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform'),
                'slots': [
                    util.build_slot_dict("b64_img_str", b64_img_str, 1.0)
                ]
            }
        ]
    }
    system.observe(observation)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.CONFIRM]

    # Turn id 3: Yes
    observation = {
        'user_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'affirm', 0.9),
            }
        ],
        'photoshop_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform'),
                'slots': [
                    util.build_slot_dict("b64_img_str", b64_img_str, 1.0)
                ]
            }
        ]
    }
    system.observe(observation)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.REQUEST_LABEL]

    # Turn id 4: dog 1
    observation = {
        'user_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform', 1.0),
                'slots': [
                    util.build_slot_dict('object_mask_id', '1', 0.5)
                ]
            }
        ],
        'photoshop_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform'),
                'slots': [
                    util.build_slot_dict("b64_img_str", b64_img_str, 1.0)
                ]
            }
        ]
    }
    system.observe(observation)
    system_act = system.act()
    assert extract_das(system_act) == [SystemAct.CONFIRM]

    # Turn id 5: No, dog 0
    observation = {
        'user_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'negate', 0.9),
            },
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform', 1.0),
                'slots': [
                    util.build_slot_dict('object_mask_id', '0', 0.75)
                ]
            }
        ],
        'photoshop_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform'),
                'slots': [
                    util.build_slot_dict("b64_img_str", b64_img_str, 1.0)
                ]
            }
        ]
    }
    system.observe(observation)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.EXECUTE, SystemAct.ASK]

    # Turn id 6: Adjust brightness
    observation = {
        'user_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform', 1.0),
                'intent': util.build_slot_dict('intent', 'adjust', 1.0),
                'slots': [
                    util.build_slot_dict('attribute', 'brightness', 0.95)
                ]
            }
        ],
        'photoshop_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform'),
                'slots': [
                    util.build_slot_dict("b64_img_str", b64_img_str, 1.0)
                ]
            }
        ]
    }
    system.observe(observation)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.REQUEST]

    # Turn id 6: Adjust brightness
    observation = {
        'user_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform', 1.0),
                'slots': [
                    util.build_slot_dict('adjust_value', 50, 0.95)
                ]
            }
        ],
        'photoshop_acts': [
            {
                'dialogue_act': util.build_slot_dict('dialogue_act', 'inform'),
                'slots': [
                    util.build_slot_dict("b64_img_str", b64_img_str, 1.0)
                ]
            }
        ]
    }
    system.observe(observation)
    system_act = system.act()

    assert extract_das(system_act) == [SystemAct.EXECUTE, SystemAct.ASK]



