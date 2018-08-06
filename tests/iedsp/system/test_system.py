import configparser


from iedsp.core import UserAct, SystemAct, Hermes
from iedsp.ontology import ImageEditOntology
from iedsp.photoshop import PhotoshopGateway
from iedsp.system import RuleBasedDialogueManager
from iedsp.user import AgendaBasedUserSimulator


def test_dialogue_flow_and_dialogue_act():
    """ Here the system parleys with user acts
    """
    config = configparser.ConfigParser()
    config.read('config.dev.ini')

    user = AgendaBasedUserSimulator(config["USER"])
    system = RuleBasedDialogueManager(config["SYSTEM"])
    photoshop = PhotoshopGateway(config["PHOTOSHOP"])
    system.reset()
    photoshop.reset()

    turn_idx = 0

    def parley(observation, system, photoshop, turn_idx):
        print("Turn", turn_idx)
        print("User", user.template_nlg(observation['user_acts']))

        photoshop_act = photoshop.act()

        system.observe(photoshop_act)
        system.observe(observation)
        system_act = system.act()

        print("System", system_act['system_utterance'])

        photoshop.observe(system_act)
        photoshop.act()
        dialogue_acts = [a['dialogue_act']['value']
                         for a in system_act['system_acts']]
        return dialogue_acts

    # Turn 1
    observation = {
        'user_acts': [
            {'dialogue_act': Hermes.build_slot_dict('dialogue_act', 'open', 1.),
             'slots': [
                 {'slot': 'image_path',
                  'value': '/Users/tzlin/Documents/code/SimplePhotoshop/images/3.jpg',
                  'conf': 1.0}
            ]
            }
        ],
        'episode_done': False,
    }

    dialogue_acts = parley(observation, system, photoshop, turn_idx)
    assert dialogue_acts == ['execute', 'greeting', 'ask']

    # Turn 2
    observation = {
        'user_acts': [
            {
                'dialogue_act': Hermes.build_slot_dict('dialogue_act', 'undo', 0.5),
                'slots': [],
            }
        ],
        'episode_done': False,
    }
    dialogue_acts = parley(observation, system, photoshop, turn_idx)
    assert dialogue_acts == ['confirm']

    # Turn 3
    observation = {
        'user_acts': [
            {
                'dialogue_act': Hermes.build_slot_dict('dialogue_act', UserAct.NEGATE, 0.9)
            }
        ]
    }
    dialogue_acts = parley(observation, system, photoshop, turn_idx)
    assert dialogue_acts == ['ask']

    # Turn 4
    observation = {
        'user_acts': [
            {
                'dialogue_act': Hermes.build_slot_dict('dialogue_act', 'adjust', 0.5),
                'slots': [
                    Hermes.build_slot_dict('attribute', 'brightness', 1.0),
                    Hermes.build_slot_dict('object', 'dog', 0.8)
                ]
            }
        ],
        'episode_done': False,
    }
    dialogue_acts = parley(observation, system, photoshop, turn_idx)
    assert dialogue_acts == ['confirm']

    # Turn 5
    observation = {
        'user_acts': [
            {
                'dialogue_act': Hermes.build_slot_dict('dialogue_act', UserAct.AFFIRM, 0.9)
            }
        ]
    }
    dialogue_acts = parley(observation, system, photoshop, turn_idx)
    assert dialogue_acts == ['request']

    # Turn 6
    observation = {
        'user_acts': [
            {
                'dialogue_act': Hermes.build_slot_dict('dialogue_act', 'adjust', 0.5),
                'slots': [
                    Hermes.build_slot_dict('attribute', 'brightness', 1.0),
                    Hermes.build_slot_dict('object', 'dog', 0.8)
                ]
            }
        ],
        'episode_done': False,
    }
    dialogue_acts = parley(observation, system, photoshop, turn_idx)

    assert dialogue_acts == ['confirm']

    # Turn 7
    observation = {
        'user_acts': [
            {
                'dialogue_act': Hermes.build_slot_dict('dialogue_act', UserAct.AFFIRM, 0.9)
            }
        ]
    }
    dialogue_acts = parley(observation, system, photoshop, turn_idx)
    assert dialogue_acts == ['request']

    import pdb
    pdb.set_trace()
