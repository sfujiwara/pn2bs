# -*- coding: utf-8 -*-

import flask
import predictor

app = flask.Flask(__name__)
# app.debug = True


@app.route('/pred', methods=['GET', 'POST'])
def main():
    poke_name = flask.request.args.get('name')
    res = predictor.predict(poke_name)
    return flask.jsonify(res)


# if __name__ == '__main__':
#     app.run()
