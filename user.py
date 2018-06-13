import copy
import random

import numpy as np

import lightroom


def generate_random_profile():
    """
        Generates a random profile for

    """
    # Params
    # Agenda related
    # Probability to sample two actions
    two_action_prob = random.uniform(0., 1.)

    # Parameter related to reforming probabilities
    temperature = random.randint(1, 5)
    estimation_error_prob = random.uniform(
        0., 1.)  # Estimation error probability

    slot_error_prob = random.uniform(0., 1.)

    profile = {"temperature": temperature,
               "estimation_error_prob": estimation_error_prob,
               "two_action_prob": two_action_prob,
               "slot_error_prob": slot_error_prob}

    return profile


class AgendaBasedUserSimulator(object):
    """
        User Simulator
    """

    def __init__(self):
        """
            agenda stores the current agenda
            global_agenda stores the remaining action list
        """
        self.agenda = list()  # Local agenda
        self.global_agenda = list()  # Global agenda
        self.threshold = 5

    @property
    def name(self):
        return self.__class__.__name__

    def display_agenda(self):
        """
            prints the agenda in human readable format
        """
        pass

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

    def _flush_global_agenda_to_agenda(self):
        """
            A goal can be
                a. 1 or more adjust slots
                b. 1 non_adjust slot
            This function pops a goal from global_agenda and inserts into
            agenda
        """
        assert len(self.agenda) == 0, "Current agenda should be empty!"

        # Treat agenda as Queues
        while len(self.global_agenda):
            slot_dict = self.global_agenda[0]  # front of queue
            slot_type = slot_dict['type']
            if slot_type == "non_adjust":
                if len(self.agenda) == 0:  # Push to agenda if agenda is empty
                    slot_dict = self.global_agenda.pop(0)
                    self.agenda.append(slot_dict)
                break
            elif slot_type == "adjust":  # Push adjust actions into agenda
                slot_dict = self.global_agenda.pop(0)
                self.agenda.append(slot_dict)
            else:
                raise ValueError("Unknown slot type: {}!".format(slot_type))

    def _build_current_goal_from_agenda(self):
        # Bulid a goal with all slot values set to default
        goal = lightroom.schema.default_goal_factory()

        # TODO: continuing slots
        for slot_dict in self.agenda:
            goal[ slot_dict['slot'] ] = slot_dict['value']

        self.current_goal = goal

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
                # Sort from low to high, since pops from back, see def act(self)
                distances.sort(key=lambda tup: abs(
                    tup['distance']))
                for slot_dist in distances:
                    user_act = {}
                    user_act['type'] = 'inform'
                    user_act['slot'] = slot_dist.get('slot')

                    dist = slot_dist.get('distance')
                    estimated_value = self.estimate_distance_to_value(dist)
                    user_act['value'] = estimated_value
                    self.agenda.append(user_act)

                # Add end dialogue act
                if not len(self.agenda):
                    self.agenda.append({'type': 'end'})

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
        # Build user utterance
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
        # Sample number of actions
        num_actions = np.random.binomial(
            1, p=self.profile.get('two_action_prob', 0.)) + 1

        num_actions = min(num_actions, len(self.agenda))
        return num_actions

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

    ##########################
    #   Estimation and NLG   #
    ##########################
    def estimate_distance_to_value(self, distance):
        """
            Convert the distance to nlg estimation.
            Can also be the direct value
            TODO: parametrize this behavior.
        """
        assert abs(distance) > self.threshold

        direction = "more" if distance > 0 else "less"
        abs_distance = abs(distance)
        if abs_distance < 2 * self.threshold:
            level = "a little "
        elif abs_distance < 5 * self.threshold:
            level = ""
        else:
            level = "a lot "
        value = level + direction
        return value

    def template_nlg(self, user_acts):

        tokens = list()
        slots = list()
        for user_act in user_acts:
            if user_act['type'] == 'inform':
                act_tokens = ['make', user_act['slot'], user_act['value']]
                if len(tokens):
                    tokens += ["and"]

                slot_start = len(tokens) + 1
                slot_end = len(tokens) + 2
                value_start = len(tokens) + 2
                value_end = len(tokens) + 3

                slot = {'slot': user_act['slot'],
                        'slot_start': slot_start, 'slot_end': slot_end,
                        'value_start': value_start, 'value_end': value_end}
                slots.append(slot)
            elif user_act['type'] == 'end':
                act_tokens = ["bye"]

            tokens += act_tokens

        text = ' '.join(tokens)
        # Build utterance object
        user_utterance_obj = {}
        user_utterance_obj['text'] = text
        user_utterance_obj['tokens'] = tokens
        user_utterance_obj['slots'] = slots
        return user_utterance_obj


if __name__ == "__main__":
    user = AgendaBasedUserSimulator()
