# -*- coding: utf-8 -*-

import flask
import predictor
import json

app = flask.Flask(__name__)
# app.debug = True


@app.route('/pred', methods=['GET'])
def main():
    poke_name = flask.request.args.get('name')
    res = predictor.predict(poke_name)
    return json.dumps(res, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    app.run()
