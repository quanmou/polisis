from flask import Flask, render_template, request
import json
import os
import sys
import tensorflow as tf

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if DIR not in sys.path:
    sys.path.append(DIR)


from src.bert.practice_clf import BertClassifier


app = Flask(__name__)

bert_ckpt1 = tf.train.latest_checkpoint(f"{DIR}/model/practice_clf/2020-07-01_00")
bert_clf = BertClassifier(init_checkpoint=bert_ckpt1)


@app.route('/segment', methods=['GET', 'POST'])
def practice():
    recv_data = request.get_data()
    res = {}
    if recv_data:
        print(recv_data.decode())
        json_re = json.loads(recv_data)
        segment = json_re['segment']
        res['categories'] = bert_clf.predict(segment)
        res['labels'] = ', '.join([str(i) for i, cat in enumerate(res['categories']) if float(cat[1]) >= 0.5])
    return json.dumps(res)


@app.route('/')
def hello_world():
    return render_template('segment_label.html')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        p = int(sys.argv[1])
    else:
        p = 9005
    app.run(host='0.0.0.0',  # 任何ip都可以访问
            port=p,       # 端口
            debug=False
            )
