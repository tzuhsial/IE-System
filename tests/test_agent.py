from iedsp.agent import RuleBasedDialogueManager


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
        elif user_dialogue_act == "tool_select":
            slots = user_act['slots']
            slot_list = []
            for slot in slots:
                slot_list.append(slot['value'])
            utt = "Selecting " + ", ".join(slot_list)
        elif user_dialogue_act == "bye":
            utt = "Bye."
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
    assert agent.state == "query_cv_engine"

    agent.state = "ask_ier"
    agent.bye()
    assert agent.state == "end_session"

    # source: confirm
    agent.state = "confirm"
    agent.highNLConf_missing()
    assert agent.state == "request"

    agent.state = "confirm"
    agent.highNLConf_noMissing()
    assert agent.state == "query_cv_engine"

    # source: request
    agent.state = 'request'
    agent.lowNLConf()
    assert agent.state == 'confirm'

    agent.state = 'request'
    agent.highNLConf_noMissing()
    assert agent.state == 'query_cv_engine'

    # source: query_cv_engine
    agent.state = 'query_cv_engine'
    agent.lowCVConf()
    assert agent.state == 'ask_user_label'

    agent.state = 'query_cv_engine'
    agent.highCVConf()
    assert agent.state == 'execute_api'

    # source: ask_user_label
    agent.state = 'ask_user_label'
    agent.hasLabel()
    assert agent.state == 'execute_api'

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
    agent.reset()

    # Turn 1
    observation = {
        'user_acts': [
            {'dialogue_act': 'open',
             'slots': [
                 {'slot': 'action_type', 'value': 'open', 'conf': 1.0},
                 {'slot': 'image_path',
                  'value': '/Users/tzlin/Documents/code/SimplePhotoshop/images/3.jpg',
                  'conf': 1.0}
             ]
             }
        ]
    }
    agent.observe(observation)
    act = agent.act()
    dialogue_acts = [a['dialogue_act'] for a in act['system_acts']]
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "ask_ier"
    assert dialogue_acts == ['execute', 'greeting', 'ask']

    # Turn 2
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'action_type', 'value': 'adjust', 'conf': 0.2},
                 {'slot': 'attribute', 'value': 'brightness', 'conf': 0.3},
                 {'slot': 'adjustValue', 'value': 'more', 'conf': 0.4},
             ]}
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "ask_ier"
    assert dialogue_act == 'repeat'

    # Turn 3
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'action_type', 'value': 'adjust', 'conf': 0.7},
                 {'slot': 'attribute', 'value': 'brightness', 'conf': 0.5},
                 {'slot': 'select', 'value': 'dog', 'conf': 0.7
                  }
             ]
             }
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "confirm"
    assert dialogue_act == 'confirm'

    # Turn 4
    observation = {
        'user_acts': [
            {'dialogue_act': 'affirm'},
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])

    assert agent.state == "confirm"
    assert dialogue_act == 'confirm'

    # Turn 5
    observation = {
        'user_acts': [
            {'dialogue_act': 'negate'},
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'attribute', 'value': 'contrast', 'conf': 0.8}
             ]}
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "confirm"
    assert dialogue_act == 'confirm'

    # Turn 6
    observation = {
        'user_acts': [
            {'dialogue_act': 'affirm'},
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "request"
    assert dialogue_act == 'request'

    # Turn 7
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'adjustValue', 'value': 'more', 'conf': 0.5}
             ]}
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "confirm"
    assert dialogue_act == 'confirm'

    # Turn 8
    observation = {
        'user_acts': [
            {'dialogue_act': 'negate'},
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'adjustValue', 'value': 'less', 'conf': 0.8},
             ]}
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "ask_user_label"
    assert dialogue_act == 'request_label'

    # Turn 9
    observation = {
        'user_acts': [
            {'dialogue_act': 'tool_select',
             'slots': [
                 {'slot': 'select', 'value': 'mask1'}
             ]
             },
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "ask_ier"
    assert dialogue_act == 'execute'

    # Turn 10
    observation = {
        'user_acts': [
            {'dialogue_act': 'inform',
             'slots': [
                 {'slot': 'action_type', 'value': 'undo', 'conf': 0.8}
             ]},
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "ask_ier"
    assert dialogue_act == 'execute'

    # Turn 10
    observation = {
        'user_acts': [
            {'dialogue_act': 'bye'},
        ]
    }

    agent.observe(observation)
    act = agent.act()
    dialogue_act = act['system_acts'][0]['dialogue_act']
    print("User", user_template_nlg(observation['user_acts']))
    print("Agent", act['system_utterance'])
    assert agent.state == "end_session"
    assert dialogue_act == 'bye'

    assert False
