import copy
import random

import numpy as np

import photoshopapi


class Profile:
    """
        profile: a list that stores parameters that affects the user behavior
            Currently, we have 
            1. temperature : affects which slot to express
            2. estimation_error_prob : affects the value of slot
    """
    size = 2

    @staticmethod
    def generate_random_profile():
        # Params
        T = 1
        estimation_error_prob = random.random()

        profile = {"temperature": T,
                   "estimation_error_prob": estimation_error_prob}

        return profile


class AgendaBasedUserSimulator(object):

    profile_size = Profile.size

    def __init__(self):
        self.agenda = list()  # Use a list to store the agenda
        self.threshold = 5

    @property
    def name(self):
        return self.__class__.__name__

    def set_profile(self, profile):
        self.profile = profile

    def set_goal(self, goal):
        self.goal = goal

    def reset(self):
        self.observation = {}

    def observe(self, observation):
        # Update observation
        self.observation.update(observation)
        self._push_actions_to_agenda_from_observation()
        return self.observation

    def _push_actions_to_agenda_from_observation(self):

        # Compute distances
        image_state = self.observation.get('image_state')
        distances = self._compute_slot_distances(
            image_state)  # Compute slot distances
        distances = [slot_dist for slot_dist in distances if abs(
            slot_dist['distance']) > self.threshold]  # Filter slots above threshold

        # Decide according to actions
        system_acts = self.observation.get(
            'system_acts', [{'type': 'inform'}])  # Dummy System Act

        for sys_act in system_acts:
            if sys_act['type'] == "inform":
                # We flush the agenda
                self.agenda.clear()
                # Sort from high to low
                distances.sort(key=lambda tup: abs(
                    tup['distance']), reverse=True)

                for slot_dist in distances:
                    user_act = {}
                    user_act['type'] = 'inform'
                    user_act['slot'] = slot_dist.get('slot')
                    user_act['value'] = slot_dist.get('distance')
                    self.agenda.append(user_act)

                # Add end dialogue act
                if not len(self.agenda):
                    self.agenda.append({'type': 'end'})

            elif sys_act['type'] == "request":
                slot = sys_act.get('slot', 'null')
                value = sys_act.get('value', 'null')
                assert not(slot == 'null' and value ==
                           'null'), "slot and value cannot both be null in request!"
                # TODO
                raise NotImplementedError

    def act(self):
        """
            Current: pop only one action from the agenda.
            TODO: number of actions depend on profile param
        """
        assert len(self.agenda)
        # Episode done
        episode_done = self.agenda[0]['type'] == 'end'
        # Get user dialogue acts
        user_dialogue_acts = list()
        for _ in range(self._sample_num_actions()):
            user_dialogue_act = self.agenda.pop()
            user_dialogue_acts.append(user_dialogue_act)
        # Build userx utterance
        user_utterance = self.template_nlg(user_dialogue_acts)

        # Build return object
        user_act = {}
        user_act['user_acts'] = user_dialogue_acts
        user_act['user_utterance'] = user_utterance
        user_act['episode_done'] = episode_done
        return user_act

    ######################################
    #  Action related private functions  #
    ######################################
    def _sample_num_actions(self):
        # 1, or 2, will use a parameter in the future
        max_num_actions = len(self.agenda)
        return 1

    def _compute_slot_distances(self, image_state):
        """
            Arguments:
            - self.goal
            - image_state
            Return:
            - distances: list of dict [{"slot": "slot_1", "distance": 500}] )
        """
        distances = list()
        for slot in self.goal.keys():
            dist = self.goal[slot] - image_state[slot]
            slot_dist = {"slot": slot, "distance": dist}
            distances.append(slot_dist)
        return distances

    def _distance_to_probs(self, distances):
        """
            Arguments:
            - distances: list of dict
            Return: 
            - probs: list of float
        """
        T = self.profile['temperature']

        Xs = np.array([abs(d['distance']) for d in distances])
        Ws = np.array([1. / T] * len(distances))
        logits = Xs * Ws

        probs = (np.exp(logits) / np.sum(np.exp(logits))).tolist()
        return probs

    #################
    #      NLG      #
    #################
    def template_nlg(self, user_acts):
        pass


if __name__ == "__main__":
    user = AgendaBasedUserSimulator()
