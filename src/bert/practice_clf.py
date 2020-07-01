import os
import sys
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if DIR not in sys.path:
    sys.path.append(DIR)

import src.bert.modeling as modeling
import src.bert.optimization as optimization
import src.bert.tokenization as tokenization


if True:
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('gpu_list', '0', 'gpu list')
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    flags.DEFINE_string(
        "input_dir", f'{DIR}/data',
        "The input data dir. Should contain the .csv files (or other data files) for the task.")

    flags.DEFINE_string(
        "training_data", f'{DIR}/data/train_segment.csv',
        "The tokenized training data. Usually formatted as InputFeatures")

    flags.DEFINE_string(
        "test_data", f'{DIR}/data/test_segment.csv',
        "The tokenized training data. Usually formatted as InputFeatures")

    flags.DEFINE_string(
        "bert_config_file", f"{DIR}/model/pretrained_model/bert_config.json",
        "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")

    flags.DEFINE_string("vocab_file", f"{DIR}/model/pretrained_model/vocab.txt",
                        "The vocabulary file that the BERT model was trained on.")

    flags.DEFINE_string(
        "init_checkpoint", f"{DIR}/model/pretrained_model/bert_model.ckpt",
        "Initial checkpoint (usually from a pre-trained BERT model).")

    flags.DEFINE_string(
        "output_dir", f'{DIR}/model/practice_clf',
        "The output directory where the model checkpoints will be written.")

    flags.DEFINE_integer(
        "max_seq_length", 512,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

    flags.DEFINE_bool("do_train", False, "Whether to run training.")
    flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
    flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")
    flags.DEFINE_integer("train_batch_size", 4, "Total batch size for training.")
    flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
    flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
    flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
    flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")
    flags.DEFINE_float("warmup_proportion", 0.1,
                       "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    flags.DEFINE_integer("save_checkpoints_steps", 10, "How often to save the model checkpoint.")


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
    def get_train_examples(self, training_data):
        """See base class."""
        jd_df = self._read_csv(training_data)
        jd_data = jd_df[['segment_content', 'category_ID']].values.tolist()
        return self.create_examples(jd_data, 'training')

    def get_test_examples(self, test_data):
        """See base class."""
        jd_df = self._read_csv(test_data)
        jd_data = jd_df[['segment_content', 'category_ID']].values.tolist()
        return self.create_examples(jd_data, 'testing')

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

    @classmethod
    def _read_csv(cls, input_file):
        """Reads csv file."""
        return pd.read_csv(input_file, dtype=object)


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


def convert_examples_to_features(examples, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, FLAGS.max_seq_length, tokenizer)
        features.append(feature)
    return features


def get_input_features(examples, tokenizer, reload, set_type):
    if reload:
        with open(os.path.join(FLAGS.input_dir, '%s_features.pkl' % set_type), 'rb') as f:
            features = pickle.load(f)
            print('Reloaded %d tokenized example' % len(features))
    else:
        features = convert_examples_to_features(examples, tokenizer)
        with open(os.path.join(FLAGS.input_dir, '%s_features.pkl' % set_type), 'wb') as f:
            pickle.dump(features, f)

    return features


def generate_batch(data, batch_size):
    n_chunk = len(data) // batch_size
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_data = data[start_index:end_index]
        batch_input_ids = [item.input_ids for item in batch_data]
        batch_input_mask = [item.input_mask for item in batch_data]
        batch_segment_ids = [item.segment_ids for item in batch_data]
        batch_label_id = [item.label_id for item in batch_data]
        yield batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_id


class BertClassifier:
    def __init__(self, init_checkpoint=FLAGS.init_checkpoint, is_training=False):
        """
        checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
        is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
        """
        self.init_checkpoint = init_checkpoint
        self.is_training = is_training
        self.learning_rate = FLAGS.learning_rate
        self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)
        self.data_processor = DataProcessor()
        self.labels = self.data_processor.get_labels()
        self.num_labels = len(self.labels)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._model_builder()
            self.sess.run(tf.initialize_all_variables())

    def _model_builder(self):
        self.ids_placeholder = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.mask_placeholder = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.segment_placeholder = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, len(self.labels)])
        self.loss, self.logits, self.sigmoid_logits = self.create_model()

        self.sess_config = tf.ConfigProto()
        self.sess_config.allow_soft_placement = True
        self.sess_config.log_device_placement = False
        self.sess_config.gpu_options.allow_growth = True
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = 1
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

    def train(self, data_file=FLAGS.training_data, reload=False, save_path=FLAGS.output_dir):
        examples = self.data_processor.get_train_examples(os.path.join(FLAGS.input_dir, data_file))
        input_features = get_input_features(examples, self.tokenizer, reload, set_type='train')
        random.shuffle(input_features)
        num_train_steps = int(len(input_features) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        with self.graph.as_default():
            train_op = optimization.create_optimizer(loss=self.loss,
                                                     init_lr=self.learning_rate,
                                                     num_train_steps=num_train_steps,
                                                     num_warmup_steps=num_warmup_steps,
                                                     use_tpu=False)
            self.sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            for epoch in range(FLAGS.num_train_epochs):
                batch = 0
                for ids, mask, segment, labels in generate_batch(input_features, batch_size=FLAGS.train_batch_size):
                    train_dict = {self.ids_placeholder: ids,
                                  self.mask_placeholder: mask,
                                  self.segment_placeholder: segment,
                                  self.labels_placeholder: labels}
                    _, train_loss = self.sess.run([train_op, self.loss], feed_dict=train_dict)
                    if batch % 10 == 0:
                        print('Epoch: %d, batch: %d, training loss: %s' % (epoch, batch, train_loss))
                    if batch % 100 == 0:
                        saver.save(self.sess, os.path.join(save_path, 'model.ckpt.' + str(epoch) + '.' + str(batch)))
                    batch += 1

    def evaluate(self, data_file=FLAGS.test_data, reload=False, threshold=0.5):
        examples = self.data_processor.get_test_examples(os.path.join(FLAGS.input_dir, data_file))
        input_features = get_input_features(examples, self.tokenizer, reload, set_type='test')
        with self.graph.as_default():
            self.sess.run(tf.initialize_all_variables())
        correct_count, label_count, predict_count = 0, 0, 0
        for ids, mask, segment, labels in generate_batch(input_features, batch_size=FLAGS.eval_batch_size):
            eval_dict = {self.ids_placeholder: ids,
                         self.mask_placeholder: mask,
                         self.segment_placeholder: segment,
                         self.labels_placeholder: labels}
            sigmoid_logits = self.sess.run(self.sigmoid_logits, feed_dict=eval_dict)
            for i in range(FLAGS.eval_batch_size):
                predict_label_idx = [i for i, logit in enumerate(sigmoid_logits[i]) if logit >= threshold]
                correct_count += sum([int(labels[i][idx] == 1.0) for idx in predict_label_idx])
                label_count += sum(labels[i])
                predict_count += len(predict_label_idx)
        precision = round(correct_count / predict_count, 3)
        recall = round(correct_count / label_count, 3)
        F1 = round(2 * precision * recall / (precision + recall), 3)
        print('precision: %s' % precision)
        print('recall: %s' % recall)
        print('F1: %s' % F1)

    def predict(self, segment=''):
        example = self.data_processor.create_examples([[segment, '0,0,0,0,0,0,0,0,0,0']], set_type='predict')
        feature = convert_single_example(0, example[0], FLAGS.max_seq_length, self.tokenizer)
        predict_dict = {self.ids_placeholder: [feature.input_ids],
                        self.mask_placeholder: [feature.input_mask],
                        self.segment_placeholder: [feature.segment_ids],
                        self.labels_placeholder: [feature.label_id]}

        sigmoid_output = self.sess.run(self.sigmoid_logits, feed_dict=predict_dict)
        res = [(label, str(sigmoid_output[0][i])) for i, label in enumerate(self.labels)]
        return res


def main(_):
    checkpoint = tf.train.latest_checkpoint(os.path.join(FLAGS.output_dir, '2020-07-01_00'))
    # clf = BertClassifier(is_training=True)
    # clf.train(reload=False, save_path=os.path.join(FLAGS.output_dir, datetime.now().strftime('%Y-%m-%d_%H')))
    clf = BertClassifier(is_training=False, init_checkpoint=checkpoint)
    clf.evaluate(reload=True)


if __name__ == '__main__':
    tf.app.run()
