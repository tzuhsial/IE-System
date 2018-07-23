from configparser import ConfigParser
import json
import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

from tqdm import tqdm

from iedsp import util
from iedsp.channel import ConfChannel as Channel
from iedsp.photoshop import PhotoshopGateway
from iedsp.system import RuleBasedDialogueManager as System
from iedsp.user import AgendaBasedUserSimulator as User
from iedsp.world import SelfPlayWorld


def main(argv):
    # Get arguments
    config_file = argv[1]

    # Configsc
    config = ConfigParser()
    config.read(config_file)

    agendas = util.load_from_pickle(config['DEFAULT']['AGENDA_PICKLE'])

    # Build Interaction World
    user = User(config['USER'])
    channel = Channel(config['CHANNEL'])
    system = System(config['SYSTEM'])
    photoshop = PhotoshopGateway(config['PHOTOSHOP'])

    agents = [user, channel, system, photoshop]

    world = SelfPlayWorld(agents)

    for agenda in tqdm(agendas):
        world.reset()
        user.load_agenda(agenda)

        while True:

            world.parley()

            if world.episode_done():
                break

        pass


if __name__ == "__main__":
    main(sys.argv)
