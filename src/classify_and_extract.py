import os
import sys
import tensorflow as tf
from src.bert.practice_clf import BertClassifier
from src.bert.extract_attribute import BertNer
from data.attributes_ner_label import attribute_infos
from data.category_attributes import category_attributes

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIR not in sys.path:
    sys.path.append(DIR)


class ClassifyAndExtract:
    def __int__(self):
        clf_ckpt = tf.train.latest_checkpoint(f"{DIR}/model/category_model/2020-07-05_13_4epoch")
        self.clf_model = BertClassifier(init_checkpoint=clf_ckpt)

    def cls_and_extract(self, policy):
        res = []
        seg_category = self.clf_model.predict_long_text(policy)
        for item in seg_category:
            segment, pred, label_idx = item
            cat = {}
            for idx in label_idx:
                category = self.clf_model.labels[idx]
                attr = {}
                for attribute in category_attributes[category]:
                    folder_name = attribute_infos[attribute]["file_prefix"]
                    checkpoint = tf.train.latest_checkpoint(f"{DIR}/model/attribute_model/{folder_name}")
                    bert_ner = BertNer(attribute, init_checkpoint=checkpoint)
                    output = bert_ner.predict(segment)
                    labels = []
                    for i, word in enumerate(segment.split(' ')):
                        if output[i + 1] == 'O' or output[i + 1] == '[PAD]':
                            labels.append(word)
                        else:
                            labels.append(word + '(' + output[i + 1] + ')')
                    attribute_output = " ".join(labels)
                    attr[attribute] = attribute_output
                cat[category] = attr
            res.append(cat)
        return res


policy = """Why and How We Collect and Store Information Sally Ride Science collects and stores some information about 
your daughter or son, to allow her or him to participate in some activities. For example, we store and use email 
addresses and other contact information to communicate with users about contests and other special events. We request 
parent contact information and parent consent when a girl or boy participates in Sally Ride Science events and 
activities. This enables us to confirm that a parent consents to their daughter's or son's participation in an event 
or activity and agrees to the terms of our privacy policy. <br> <br>
"""
cae = ClassifyAndExtract()
result = cae.cls_and_extract(policy)
