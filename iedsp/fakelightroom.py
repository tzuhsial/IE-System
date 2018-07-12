"""
    An interface to build photoshop related apis.
    This file also includes the schema used
"""
import copy


class schema:
    """
        Simple Class to define schema
    """
    slot_types = ["adjust", "non_adjust"]
    ####################
    #   Adjust Slots   #
    ####################
    adjust_slots = ["adjust_slot_1", "adjust_slot_2", "adjust_slot_3"]
    adjust_slots_range = {
        "adjust_slot_1": {"min": -100, "max": 100},
        "adjust_slot_2": {"min": -50, "max":  50},
        "adjust_slot_3": {"min": -10, "max":  10},
    }

    ########################
    #   Non Adjust Slots   #
    ########################
    # Set first option as default

    non_adjust_slots = ["non_adjust_slot_1", "non_adjust_slot_2"]
    non_adjust_slots_options = {
        "non_adjust_slot_1": ["non_adjust_slot_1_option1", "non_adjust_slot_1_option2"],
        "non_adjust_slot_2": ["non_adjust_slot_2_option1", "non_adjust_slot_2_option2", "non_adjust_slot_2_option3"],
    }

    @staticmethod
    def default_goal_factory():
        """
            Returns a goal with all slot values set to default
            adjust_slot: 0.
            non_adjust_slot: default_option
        """
        default_goal = {}
        for adjust_slot in schema.adjust_slots:
            default_goal[adjust_slot] = 0.
        for non_adjust_slot in schema.non_adjust_slots:
            default_option = schema.non_adjust_slots_options[non_adjust_slot][0]
            default_goal[non_adjust_slot] = default_option
        return default_goal


class LightroomAPI(object):
    def __init__(self):
        """
            TODO: connect with Lightroom
        """
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        self._status = schema.default_goal_factory()
        self.observe({})

    @property
    def status(self):
        """
            Json Representation
        """
        self._status

    def observe(self, observation):
        self.observation = observation

    def act(self):
        """
            Execute action on the image,
            action_type: 
                1. adjust
                2. non_adjust
        """
        # Get all system actions
        system_acts = self.observation.get('system_acts', list())
        for sys_act in system_acts:

            # Get parameters
            dialogue_act_type = sys_act.get('type')
            action_type = sys_act.get('action_type', 'null')
            slot_type = sys_act.get('type')
            slot_name = sys_act.get('slot')
            slot_value = sys_act.get('value')

            # Execute actions
            if action_type == 'null':
                pass
            elif dialogue_act_type == "inform":
                if slot_type == "adjust":
                    self._status[slot_name] += slot_value
                elif slot_type == "non_adjust":
                    self._status[slot_name] = slot_value
            else:
                raise NotImplementedError(
                    "Unknown dialogue_act_type \"{}\"".format(dialogue_act_type))

        photoshop_act = {}
        photoshop_act['image_state'] = copy.deepcopy(self._status)
        return photoshop_act
