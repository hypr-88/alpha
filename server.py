from copy import deepcopy

from flask import Flask, request, jsonify
from rq import Queue
from redis import Redis
from rq.registry import StartedJobRegistry

from Config import deriveFeature, basicFeature, transform
from PreFetch import getData
from task import startAlphaEvolve

app = Flask(__name__)
redis_conn = Redis()
rq = Queue('default', connection=redis_conn)
registry = StartedJobRegistry('default', connection=redis_conn)


@app.route("/mine", methods=['POST'])
def mineAlpha():
    content = request.json
    if 'basic' not in content or 'derive' not in content and 'params' not in content:
        return jsonify({"error": "Misisng input, basic, derive, params", "data": None})

    basic = content['basic']
    derive = content['derive']
    params = content['params']
    data = getData()

    featureCols = []
    featureCols.extend(deriveFeature(derive, data))
    featureCols.extend(basicFeature(basic, data))
    transformTable = None

    if 'normalized' in content:
        transformTable = transform(data)

    # serialize df before sendind to rq
    for symbol in data.keys():
        data[symbol] = data[symbol].to_json(orient="index")

    alphaData = deepcopy(params)
    alphaData['featureList'] = featureCols
    alphaData['data'] = data

    # if len(registry.get_job_ids()) > 0:
    #     return jsonify({"data": None, "error": "Another job is being executed"})
    #
    # rq.enqueue(startAlphaEvolve, alphaData)

    return jsonify({"data": {
        "columns": featureCols,
        "params": params
    }, "error": None})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
