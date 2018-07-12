import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    add_basic_opt(parser)
    add_photoshop_opt(parser)
    add_user_opt(parser)
    add_agent_opt(parser)
    args = parser.parse_args()
    opt = vars(args)
    return opt


def add_basic_opt(parser):
    parser.add_argument('-scenario_pickle', dest='scenario_pickle',
                        type=str, default='scenario.photoshop.pickle')
    parser.add_argument('-api_url',dest='api_url',type=str,default='http://localhost:5000')
    return parser


def add_photoshop_opt(parser):
    return parser

def add_user_opt(parser):
    parser.add_argument('-user_adjust_threshold', type=int, default=5)
    return parser

def add_agent_opt(parser):
    return parser
