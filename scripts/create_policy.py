import sys

import numpy as np

from iedsp.ontology import OntologyEngine
from iedsp.policy import ActionMap, builder as policylib
from iedsp.system import State
import iedsp.util as util


config_file = sys.argv[1]
config = util.load_from_json(config_file)

# Policy name
policy_name = config["agents"]["system"]["policy"]

# Policy init args
policy_config = config["policy"]
ontology_json = util.load_from_json(config["ontology"])

ontology = OntologyEngine(ontology_json)
state = State(ontology)

policy_config["qnetwork"]["input_size"] = len(state.to_list())
action_mapper = ActionMap(ontology_json)
policy_config["qnetwork"]["output_size"] = action_mapper.size()

policy = policylib(policy_name)(policy_config, action_mapper)


policy.load('./exp/tmp.ckpt')
state = np.random.random(101).tolist()

action_idx = policy.step(state)


policy.save("./exp/tmp.ckpt")
