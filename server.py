from flask import Flask
from flask_rq2 import RQ

app = Flask(__name__)
rq = RQ(app)


@app.route("/", methods=['POST'])
def index():
    return "Hello World!"
