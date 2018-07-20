import copy
import random

import numpy as np


class AgendaBasedUserSimulator(object):
    """
        Agenda Based User Simulator: Photoshop actions
    """

    def __init__(self):
        """
            agenda stores the current agenda
            global_agenda stores the remaining action list
        """
        self.agenda = list()  # Local agenda
        self.adjust_threshold = opt['user_adjust_threshold']

    @property
    def name(self):
        return self.__class__.__name__

    def print_agenda(self):
        """
            prints the agenda in human readable format
        """
        def act_to_str(act):
            string = "| {: >15} | {: >20} | {: >25} |"\
                .format(act['type'], act['slot'], act['value'])
            return string

        print("-" * 70)
        print("|" + " " * 31 + "Agenda" + " " * 31 + "|")
        print("-" * 70)
        for act in self.agenda:
            print(act_to_str(act))
        print("-" * 70)

        print("-" * 70)
        print("|" + " " * 27 + "Global Agenda" + " " * 28 + "|")
        print("-" * 70)
        for act in self.global_agenda:
            print(act_to_str(act))
        print("-" * 70)

    def setup_scenario(self, profile, multi_goal):
        """
            Scenario consists of a profile and a multi_goal
                profile: list of floats
                multi_goal: list of dicts
        """
        self.profile = profile
        self.global_agenda = multi_goal

        # This is the tricky part...
        self._flush_global_agenda_to_agenda()
        self._build_current_goal_from_agenda()

    def reset(self):
        self.agenda = list()
        self.global_agenda = list()

        self.observation = lightroom.schema.default_goal_factory()
        self.goal = {}

    def observe(self, observation):
        self.observation.update(observation)
        self._flush_global_agenda_to_agenda()
        self._build_current_goal_from_agenda()
        return self.observation

    def act(self):
        """
            Current: pop only one action from the agenda.
            TODO: number of actions depend on profile param
        """
        assert len(self.agenda)
        # Episode done
        episode_done = (self.agenda[0]['type'] == 'end')

        # Get user dialogue acts
        user_dialogue_acts = list()
        for _ in range(self._sample_num_actions()):
            user_dialogue_act = self.agenda.pop(0)
            user_dialogue_acts.append(user_dialogue_act)

        # Build user utterance
        user_utterance = self.template_nlg(user_dialogue_acts)

        # Build return object
        user_act = {}
        user_act['user_acts'] = user_dialogue_acts
        user_act['user_utterance'] = user_utterance
        user_act['episode_done'] = episode_done
        return user_act


if __name__ == "__main__":
    user = AgendaBasedUserSimulator()
