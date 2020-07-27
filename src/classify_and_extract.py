import os
import sys
import tensorflow as tf
from src.bert.extract_attribute import BertNer
from data.attributes_ner_label import attribute_infos
from data.category_attributes import category_attributes

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIR not in sys.path:
    sys.path.append(DIR)

import src.bert.modeling as modeling
import src.bert.tokenization as tokenization


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """See base class."""
        return ["First Party Collection/Use", "Third Party Sharing/Collection", "User Choice/Control",
                "User Access, Edit and Deletion", "Data Retention", "Data Security", "Policy Change",
                "Do Not Track", "International and Specific Audiences", "Other"]

    @classmethod
    def create_examples(cls, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(str(i+1)))
            text_a = tokenization.convert_to_unicode(str(line[0]))
            text_b = tokenization.convert_to_unicode('')
            label = list(map(float, line[1].split(',')))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s" % example.label)

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=example.label)
  return feature


class BertClassifier:
    def __init__(self, init_checkpoint, is_training=False):
        """
        checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
        is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
        """
        self.init_checkpoint = init_checkpoint
        self.is_training = is_training
        self.bert_config = modeling.BertConfig.from_json_file(f"{DIR}/model/pretrained_model/bert_config.json")
        self.tokenizer = tokenization.FullTokenizer(vocab_file=f"{DIR}/model/pretrained_model/vocab.txt")
        self.data_processor = DataProcessor()
        self.labels = self.data_processor.get_labels()
        self.num_labels = len(self.labels)
        self.max_seq_length = 256
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._model_builder()
            self.sess.run(tf.initialize_all_variables())

    def _model_builder(self):
        self.ids_placeholder = tf.placeholder(tf.int32, shape=[None, self.max_seq_length])
        self.mask_placeholder = tf.placeholder(tf.int32, shape=[None, self.max_seq_length])
        self.segment_placeholder = tf.placeholder(tf.int32, shape=[None, self.max_seq_length])
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_labels])
        self.loss, self.logits, self.sigmoid_logits = self.create_model()

        self.sess_config = tf.ConfigProto()
        self.sess_config.allow_soft_placement = True
        self.sess_config.log_device_placement = False
        self.sess_config.gpu_options.allow_growth = True
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(config=self.sess_config)
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        print("restore from the checkpoint {0}".format(self.init_checkpoint))

    def create_model(self, is_training=False, use_one_hot_embedding=True):
        """Create a classification moodel."""
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=self.ids_placeholder,
            input_mask=self.mask_placeholder,
            token_type_ids=self.segment_placeholder,
            use_one_hot_embeddings=use_one_hot_embedding
        )
        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable("output_bias", [self.num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if self.is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            label_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder, logits=logits)
            loss = tf.reduce_mean(label_loss)
            sigmoid_logits = tf.math.sigmoid(logits)
        return loss, logits, sigmoid_logits

    def predict(self, segment=''):
        example = self.data_processor.create_examples([[segment, '0,0,0,0,0,0,0,0,0,0']], set_type='predict')
        feature = convert_single_example(0, example[0], self.max_seq_length, self.tokenizer)
        predict_dict = {self.ids_placeholder: [feature.input_ids],
                        self.mask_placeholder: [feature.input_mask],
                        self.segment_placeholder: [feature.segment_ids],
                        self.labels_placeholder: [feature.label_id]}

        sigmoid_output = self.sess.run(self.sigmoid_logits, feed_dict=predict_dict)
        res = [(i, label, sigmoid_output[0][i]) for i, label in enumerate(self.labels)]
        return res

    def predict_long_text(self, text):
        segments = text.split('|||')
        res = []
        for seg in segments:
            if seg:
                pred = self.predict(seg)
                label_idx = [i for i, p in enumerate(pred) if float(p[1]) >= 0.5]
                res.append([seg, pred, label_idx])
        return res


class ClassifyAndExtract:
    def __init__(self):
        clf_ckpt = tf.train.latest_checkpoint(f"{DIR}/model/category_model/2020-07-05_13_4epoch")
        self.clf_model = BertClassifier(init_checkpoint=clf_ckpt)

    def cls_and_extract(self, segment):
        res = []
        seg_category = self.clf_model.predict(segment)
        for item in seg_category:
            idx, category, prob = item
            cat = {}
            if prob >= 0.90:
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

    def cls_and_extract_for_long_text(self, text):
        segments = text.split('|||')
        res = []
        for i, seg in enumerate(segments):
            if seg:
                output = self.cls_and_extract(seg)
                res.append([i, seg, output])
        return res


policy = """In some areas of our websites, Jobscan requests or may request that you provide personal information, including your name, gender, address, email address, telephone number, contact information, birthdate, billing information and any other information from which your identity is discernible. Jobscan may also request that you upload a copy of your resume and connect Jobscan to your LinkedIn profile, thereby providing personal information regarding your education, prior employment, occupation, titles, and skills, as well as your most recent profile picture, the last article you may have shares, number of connections, industry information and non-specific geolocation information.
|||
We also gather certain information about your use of our site, such as what areas you visit and what services you access. Moreover, there is information about your computer hardware and software that is collected by Jobscan. This information can include without limitation your IP address, browser type, domain names, access times and referring website addresses. We gather this information automatically and store it in log files. We do not link this automatically collected date to other information we collect about you.
|||
Jobscan also collects, or logs, certain other information that cannot identify you personally when you visit our Sites. This information includes your Internet Protocol ("IP") address and your domain name. An IP address is a number that is automatically assigned to your computer by the ISP computer through which you access the web and a domain name is the name of the ISP computer itself through which you access the web. Jobscan logs these IP addresses and domain names and aggregates them for system administration and to monitor the use of our site. We may use the aggregated information to measure the number of visits to our site, the average time spent on our site, the number of pages viewed, and various other site use statistics. Such monitoring helps us evaluate how our Sites are used and continuously improve the content we provide."""

cae = ClassifyAndExtract()
result = cae.cls_and_extract_for_long_text(policy)
