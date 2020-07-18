import os
import sys
import json
from flask import Flask, render_template, request
import tensorflow as tf

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if DIR not in sys.path:
    sys.path.append(DIR)

from src.bert.extract_attribute import BertNer

app = Flask(__name__)

checkpoint = tf.train.latest_checkpoint(f"{DIR}/model/attribute_model/2020-07-18_21_4epoch")
bert_ner = BertNer(init_checkpoint=checkpoint)


@app.route('/segment', methods=['GET', 'POST'])
def practice():
    recv_data = request.get_data()
    res = {}
    if recv_data:
        print(recv_data.decode())
        json_re = json.loads(recv_data)
        segment = json_re['segment']
        output = bert_ner.predict(segment)
        labels = []
        for i, word in enumerate(segment.split(' ')):
            if output[i+1] == 'O' or output[i+1] == '[PAD]':
                labels.append(word)
            else:
                labels.append(word + '(' + output[i+1] + ')')
        res['output'] = " ".join(labels)
    return json.dumps(res)


@app.route('/')
def hello_world():
    return render_template('extract_attribute.html')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        p = int(sys.argv[1])
    else:
        p = 9005
    app.run(host='0.0.0.0',  # 任何ip都可以访问
            port=p,          # 端口
            debug=False
            )
