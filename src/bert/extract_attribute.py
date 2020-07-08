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
        "data_dir", f'{DIR}/data',
        "The input data dir. Should contain the .csv files (or other data files) for the task.")

    flags.DEFINE_string(
        "training_data", f'{DIR}/data/train_segment.csv',
        "The tokenized training data. Usually formatted as InputFeatures")

    flags.DEFINE_string(
        "validation_data", f'{DIR}/data/validation_segment.csv',
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

    flags.DEFINE_bool(
        "do_lower_case", False,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

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
    flags.DEFINE_integer("save_checkpoints_steps", 100, "How often to save the model checkpoint.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               mask,
               segment_ids,
               label_ids):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls,input_file):
        """Read a BIO data!"""
        rf = open(input_file,'r')
        lines = [];words = [];labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if len(line.strip())==0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l,w))
                words=[]
                labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        return ["[PAD]","B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature
    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]
    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.middle_output+"/label2id.pkl",'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1]*len(input_ids)
    #use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature,ntokens,label_ids


# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    # TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)
    return loss, transition


def softmax_layer(logits,labels,num_labels,mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask,dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict

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


class BertNer:
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
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_labels])
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

    def create_model(self, bert_config, is_training, input_ids, mask,
                     segment_ids, labels, num_labels, use_one_hot_embeddings):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        output_layer = model.get_sequence_output()
        # output_layer shape is
        if is_training:
            output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
        logits = hidden2tag(output_layer, num_labels)
        # TODO test shape
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        if FLAGS.crf:
            mask2len = tf.reduce_sum(mask, axis=1)
            loss, trans = crf_loss(logits, labels, mask, num_labels, mask2len)
            predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
            return (loss, logits, predict)

        else:
            loss, predict = softmax_layer(logits, labels, num_labels, mask)

            return (loss, logits, predict)

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
            saver = tf.train.Saver(max_to_keep=2, save_relative_paths=True)
            save_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H') + '_' + str(FLAGS.num_train_epochs) + 'epoch')
            for epoch in range(FLAGS.num_train_epochs):
                batch, correct_count, label_count, predict_count = 0, 0, 0, 0
                for ids, mask, segment, labels in generate_batch(input_features, batch_size=FLAGS.train_batch_size):
                    train_dict = {self.ids_placeholder: ids,
                                  self.mask_placeholder: mask,
                                  self.segment_placeholder: segment,
                                  self.labels_placeholder: labels}
                    _, train_loss, sigmoid_logits = self.sess.run([train_op, self.loss, self.sigmoid_logits], feed_dict=train_dict)
                    # 统计训练准召
                    for i in range(FLAGS.train_batch_size):
                        predict_label_idx = [i for i, logit in enumerate(sigmoid_logits[i]) if logit >= 0.5]
                        correct_count += sum([int(labels[i][idx] == 1.0) for idx in predict_label_idx])
                        label_count += sum(labels[i])
                        predict_count += len(predict_label_idx)
                    precision = round(correct_count / predict_count, 3) if predict_count != 0 else 0.0
                    recall = round(correct_count / label_count, 3) if label_count != 0 else 0.0
                    F1 = round(2 * precision * recall / (precision + recall), 3) if (predict_count != 0 and label_count != 0) else 0.0
                    if batch % 10 == 0:
                        print('Epoch: %d, batch: %d, training loss: %s, precision: %s, recall: %s, F1: %s'
                              % (epoch, batch, train_loss, precision, recall, F1))

                    if batch % 100 == 0:
                        saver.save(self.sess, os.path.join(save_path, 'model.ckpt.' + str(epoch) + '.' + str(batch)))
                    batch += 1


def main(_):
    pass


if __name__ == "__main__":
    tf.app.run()