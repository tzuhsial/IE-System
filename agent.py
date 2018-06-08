"""
    Finite State Machine Agent
"""

from transitions import Machine


class FSMBase(object):
    """
        Separate FSMBase from Agent for abstraction
    """
    states = [
        'start_session', 'ask_ier', 'request_missing', 'execute_action',
        'end_session'
    ]

    transitions = [{
        'trigger': 'greeting',
        'source': 'start_session',
        'dest': 'ask_ier'
    }, {
        'trigger': 'missing',
        'source': 'ask_ier',
        'dest': 'request_missing'
    }, {
        'trigger': 'missing',
        'source': 'request_missing',
        'dest': 'request_missing'
    }, {
        'trigger': 'no_missing',
        'source': 'ask_ier',
        'dest': 'execute_action'
    }, {
        'trigger': 'no_missing',
        'source': 'request_missing',
        'dest': 'execute_action'
    }, {
        'trigger': 'show_result',
        'source': 'execute_action',
        'dest': 'ask_ier'
    }, {
        'trigger': 'no_request',
        'source': 'ask_ier',
        'dest': 'end_session'
    }]
    # Agent Dialogue Acts
    acts = ['welcome', 'inform', 'request']

    def __init__(self):
        self.machine = Machine(model=self, states=self.states,
                               transitions=self.transitions, initial='start_session')


class Agent(FSMBase):
    """
        A finite state machine based agent
    """
    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        # Initialize starting state
        self.state = 'start_session'
        # Initialize observation
        self.observation = {}
        self.greeting()

    def observe(self, observation):
        self.observation.update(observation)
        return self.observation

    def act(self):
        # Get params from observation
        user_acts = self.observation.get('user_acts')
        user_utterance = self.observation.get('user_utterance', '')
        episode_done = False

        system_acts = list()
        for user_act in user_acts:
            if user_act['type'] == 'inform':
                sys_act = {}
                sys_act['type'] = 'inform'
                sys_act['action_type'] = 'adjust'
                sys_act['slot'] = user_act.get('slot')
                sys_act['value'] = user_act.get('value')
            elif user_act['type'] == 'end':
                episode_done = True
                sys_act = {}
                sys_act['type'] = 'bye'
            else:
                raise NotImplementedError
            system_acts.append(sys_act)

        system_utterance_obj = self.template_nlg(system_acts)

        agent_act = {}
        agent_act['system_acts'] = system_acts
        agent_act['system_utterance'] = system_utterance_obj
        agent_act['episode_done'] = episode_done

        return agent_act

    def template_nlu(self):
        """
            Simple natural language understanding for template language
        """
        pass

    def template_nlg(self, system_acts):
        """
            Templated based natural language generation for crowd source paraphrasing
        """
        return None

    ###################################
    #  State Transition Dialogue Act  #
    ###################################
    def on_enter_ask_ier(self):
        agent_act = {}
        agent_act['agent_dialogue_act'] = 'bye'
        agent_act['message'] = 'bye'
        agent_act['action_type'] = agent_act['slot'] = agent_act['value'] = 'null'
        agent_act['episode_done'] = True
        self.agent_act = agent_act

    def on_enter_request_missing(self):
        slot = self.observation['slot']
        value = self.observation['value']

        agent_act = {}
        agent_act['agent_dialogue_act'] = 'bye'
        agent_act['message'] = 'bye'
        agent_act['action_type'] = agent_act['slot'] = agent_act['value'] = 'null'
        agent_act['episode_done'] = True
        self.agent_act = agent_act

    def on_enter_execute_action(self):
        """
            Perform Simple NLU on the user messages
        """
        user_utterance = self.observation['']
        pass

    def on_enter_end_session(self):
        agent_act = {}
        agent_act['agent_dialogue_act'] = 'bye'
        agent_act['message'] = 'bye'
        agent_act['action_type'] = agent_act['slot'] = agent_act['value'] = 'null'
        agent_act['episode_done'] = True
        self.agent_act = agent_act


if __name__ == "__main__":
    agent = Agent()
    agent.greeting()
    agent.no_request()
    import pdb
    pdb.set_trace()
