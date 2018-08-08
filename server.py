import argparse
import configparser
import json


from flask import Flask, request, jsonify, render_template

from iedsp.system import System
from iedsp.photoshop import PhotoshopPortal


config = configparser.ConfigParser()
config.read('config.dev.ini')

system = System(config)
photoshop = PhotoshopPortal(config["PHOTOSHOP"])

app = Flask(
    __name__, template_folder='./app/template', static_folder='./app/static')


@app.route('/')
def index():
    system.reset()
    photoshop.reset()
    return render_template('index.html')


@app.route('/ier', methods=["POST"])
def image_edit_request():
    user_act = json.loads(request.form.get('observation'))

    system.observe(user_act)
    system_act = system.act()

    photoshop.observe(system_act)
    photoshop_act = photoshop.act()

    system.observe(photoshop_act)
    # Build return object
    obj = {**system_act, **photoshop_act}
    return jsonify(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=2003)
    parser.add_argument('-d', '--debug', action="store_true")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
