import argparse
import json

from flask import Flask, request, jsonify, render_template

import sps.utils as utils
from sps import SimplePhotoshop

ps = SimplePhotoshop()

app = Flask(
    __name__, template_folder='./app/template', static_folder='./app/static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check', methods=["POST"])
def check():
    """Loads image if image is present
    """
    intent_type = request.form.get('intent')
    assert intent_type == "check"

    obj = {}
    if ps.getImage() is not None:
        obj['b64_img_str'] = utils.img_to_b64(ps.getImage())
    
    return jsonify(obj)


@app.route('/edit', methods=["POST"])
def edit():
    edit_type = request.form.get('edit_type')
    args = json.loads(request.form.get('args'))

    if 'adjustValue' in args:
        args['adjustValue'] = int(args['adjustValue'])  # Convert to integer

    # Get and update image
    result = ps.execute(edit_type, args)

    # Build return image object
    img = ps.getImage()
    b64_img_str = utils.img_to_b64(img)

    obj = {}
    obj['b64_img_str'] = b64_img_str
    obj['result'] = result
    return jsonify(obj)


@app.route('/control', methods=["POST"])
def control():
    control_type = request.form.get('control_type')
    args = json.loads(request.form.get('args'))
    
    # Determine on control_type, perform actions
    result = ps.control(control_type, args)

    # Load
    img = ps.getImage()
    b64_img_str = utils.img_to_b64(img)

    obj = {}
    obj['b64_img_str'] = b64_img_str
    obj['result'] = result
    return jsonify(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=2005)
    parser.add_argument('-d', '--debug', action="store_true")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
