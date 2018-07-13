import argparse
import json

from flask import Flask, request, jsonify, render_template

from iedsp.agent import RuleBasedDialogueManager
from iedsp.photoshop import SimplePhotoshopAPI

agent = RuleBasedDialogueManager()
backend = SimplePhotoshopAPI()

app = Flask(
    __name__, template_folder='./app/template', static_folder='./app/static')


@app.route('/')
def index():
    agent.reset()
    return render_template('index.html')


@app.route('/ier', methods=["POST"])
def image_edit_request():

    observation = json.loads(request.form.get('observation'))

    agent.observe(observation)
    agent_act = agent.act()

    backend.observe(agent_act)
    backend_act = backend.act()

    # Build return object
    obj = backend_act
    return jsonify(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=2003)
    parser.add_argument('-d', '--debug', action="store_true")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
