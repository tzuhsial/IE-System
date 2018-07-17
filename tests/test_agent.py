from iedsp.agent import RuleBasedDialogueManager
from iedsp.photoshop import SimplePhotoshopAPI


def user_template_nlg(user_acts):
    utt_list = []
    for user_act in user_acts:
        user_dialogue_act = user_act['dialogue_act']
        if user_dialogue_act == "open":
            utt = "(Open an image)"
        elif user_dialogue_act == "inform":
            slots = user_act['slots']
            slot_list = []
            for slot in slots:
                slot_list.append(slot['slot'] + " to be " + slot['value'])
            utt = "I want " + ', '.join(slot_list) + "."
        elif user_dialogue_act == "affirm":
            utt = "Yes."
        elif user_dialogue_act == "negate":
            utt = "No."
        elif user_dialogue_act == "select_object":
            slots = user_act['slots']
            slot_list = []
            for slot in slots:
                slot_list.append(slot['value'])
            utt = "I want to select " + ", ".join(slot_list)
        elif user_dialogue_act == "bye":
            utt = "Bye."
        elif user_dialogue_act == "select_object_mask_id":
            slot = user_act['slots'][0]
            mask_id = slot['value']
            utt = "I want to select object {}.".format(mask_id)
        else:
            raise ValueError(
                "Unknown user_dialogue_act: {}".format(user_dialogue_act))
        utt_list.append(utt)

    utterance = ' '.join(utt_list)
    return utterance


def test_state_transitions():
    """Tests state transitions(triggers) in finite state machine
    """
    # source: start_session
    agent = RuleBasedDialogueManager()
    assert agent.state == "start_session"

    agent.reset()
    assert agent.state == "start_session"

    agent.greeting()
    assert agent.state == "ask_ier"

    # source: ask_ier
    agent.state = "ask_ier"
    agent.outOfDomain()
    assert agent.state == "ask_ier"

    agent.state = "ask_ier"
    agent.zeroNLConf()
    assert agent.state == "ask_ier"

    agent.state = "ask_ier"
    agent.lowNLConf()
    assert agent.state == "confirm"

    agent.state = "ask_ier"
    agent.highNLConf_missing()
    assert agent.state == "request"

    agent.state = "ask_ier"
    agent.highNLConf_noMissing()
    assert agent.state == "ier_complete"

    agent.state = "ask_ier"
    agent.bye()
    assert agent.state == "end_session"

    # source: confirm
    agent.state = "confirm"
    agent.highNLConf_missing()
    assert agent.state == "request"

    agent.state = "confirm"
    agent.highNLConf_noMissing()
    assert agent.state == "ier_complete"

    # source: request
    agent.state = 'request'
    agent.lowNLConf()
    assert agent.state == 'confirm'

    agent.state = 'request'
    agent.highNLConf_noMissing()
    assert agent.state == 'ier_complete'

    # source: ier_complete
    agent.state = 'ier_complete'
    agent.maskMissing()
    assert agent.state == 'query_cv_engine'

    agent.state = 'ier_complete'
    agent.noMaskMissing()
    assert agent.state == 'execute_api'

    # source: query_cv_engine
    agent.state = 'query_cv_engine'
    agent.lowCVConf()
    assert agent.state == 'ask_user_label'

    agent.state = 'query_cv_engine'
    agent.highCVConf()
    assert agent.state == 'ier_complete'

    # source: ask_user_label
    agent.state = 'ask_user_label'
    agent.hasLabel()
    assert agent.state == 'ier_complete'

    agent.state = 'ask_user_label'
    agent.noLabel()
    assert agent.state == 'ask_ier'

    # source: execute_api
    agent.state = 'execute_api'
    agent.result()
    assert agent.state == 'ask_ier'


def test_dialogue_flow_and_dialogue_act():
    """ Here the agent interacts with user acts
    """

    agent = RuleBasedDialogueManager()
    photoshop = SimplePhotoshopAPI()
    agent.reset()
    photoshop.reset()

    def interact(observation, agent, photoshop):
        print("User", user_template_nlg(observation['user_acts']))

        photoshop_act = photoshop.act()

        agent.observe(photoshop_act)
        agent.observe(observation)
        agent_act = agent.act()

        print("Agent", agent_act['system_utterance'])

        photoshop.observe(agent_act)
        photoshop.act()
        dialogue_acts = [a['dialogue_act'] for a in agent_act['system_acts']]
        return dialogue_acts, agent.state

    # Turn 1
    observation = {
        'user_acts': [
            {'dialogue_act': 'open',
             'slots': [
                 {'slot': 'intent', 'value': 'open', 'conf': 1.0},
                 {'slot': 'image_path',
                  'value': '/Users/tzlin/Documents/code/SimplePhotoshop/images/3.jpg',
                  'conf': 1.0}
             ]
             }
        ]
    }

    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['execute', 'greeting', 'ask']
    assert agent_state == "ask_ier"

    # Turn 2
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': []}
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['repeat']
    assert agent_state == "ask_ier"

    # Turn 3
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'intent', 'value': 'select_object', 'conf': 1.0},
                 {'slot': 'object', 'value': 'dog', 'conf': 0.7},
             ]
             }
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['confirm']
    assert agent_state == "confirm"

    # Turn 4
    observation = {
        'user_acts': [
            {'dialogue_act': 'affirm'}
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['request_label']
    assert agent_state == "ask_user_label"

    # Turn 5
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'intent', 'value': 'select_object_mask_id', 'conf': 1.0},
                 {'slot': 'object_mask_id', 'value': "1", 'conf': 1.0}
             ]
             },
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['execute', 'ask']
    assert agent_state == "ask_ier"

    # Turn 6
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'intent', 'value': 'adjust', 'conf': 0.9},
                 {'slot': 'attribute', 'value': 'brightness', 'conf': 0.7},
             ]
             }
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['confirm']
    assert agent_state == "confirm"

    # Turn 7
    observation = {
        'user_acts': [
            {'dialogue_act': 'negate'},
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'attribute', 'value': 'contrast', 'conf': 0.8}
             ]}
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['request']
    assert agent_state == "request"

    # Turn 8
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'adjustValue', 'value': '30', 'conf': 0.8}
             ]}
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['execute', 'ask']
    assert agent_state == "ask_ier"

    # Turn 9
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'intent', 'value': 'undo', 'conf': 0.8}
             ]},
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['execute', 'ask']
    assert agent_state == "ask_ier"

    # Turn 10
    observation = {
        'user_acts': [
            {'dialogue_act': 'bye'},
        ]
    }
    dialogue_acts, agent_state = interact(observation, agent, photoshop)
    assert dialogue_acts == ['bye']
    assert agent_state == "end_session"
