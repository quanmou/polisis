import os
import sys
import pandas as pd
import random
import tensorflow as tf
import pickle
from datetime import datetime

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if DIR not in sys.path:
    sys.path.append(DIR)

import src.bert.modeling as modeling
import src.bert.optimization as optimization
import src.bert.tokenization as tokenization
import data.attributes_ner_label as ner_label

if True:
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('gpu_list', '0', 'gpu list')
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    flags.DEFINE_string(
        "data_dir", f'{DIR}/data/',
        "The input data dir. Should contain the .csv files (or other data files) for the task.")

    flags.DEFINE_string(
        "model_dir", f'{DIR}/model/', "model dir.")

    flags.DEFINE_string(
        "training_data", f'{DIR}/data/attribute/pers_info_type_train.csv',
        "The tokenized training data. Usually formatted as InputFeatures")

    flags.DEFINE_string(
        "test_data", f'{DIR}/data/attribute/pers_info_type_test.csv',
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
        "output_dir", f'{DIR}/model/attribute_model',
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
    flags.DEFINE_integer("train_batch_size", 2, "Total batch size for training.")
    flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
    flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
    flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
    flags.DEFINE_integer("num_train_epochs", 4, "Total number of training epochs to perform.")
    flags.DEFINE_float("warmup_proportion", 0.1,
                       "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    flags.DEFINE_integer("save_checkpoints_steps", 100, "How often to save the model checkpoint.")
    flags.DEFINE_bool("crf", True, "use crf!")


class InputExample(object):
  """A single training/test example for simple sequence classification."""
  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text: string. The untokenized text of the first sequence. For single
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
               input_mask,
               segment_ids,
               label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
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
    def _read_data(cls, input_file):
        """Read a BIO data!"""
        df = pd.read_csv(input_file)
        lines = []
        for row in df.iterrows():
            seg, attr = row[1][0], row[1][1]
            words = seg.split(" ")
            words_offset = {}
            offset = 0
            for i, w in enumerate(words):
                words_offset.setdefault(i + offset, 'O')
                offset += len(w)

            attr = attr.split('║')
            for ar in attr:
                s, e, txt, val = ar.split('☀')
                s, e, offset = int(s), int(e), 0
                for i, w in enumerate(txt.split(' ')):
                    if i == 0:
                        words_offset[s + i + offset] = ner_label.pers_info_ner_dict[val][0]
                    else:
                        words_offset[s + i + offset] = ner_label.pers_info_ner_dict[val][1]
                    offset += len(w)

            labels = " ".join(words_offset.values())
            lines.append((seg, labels))

        return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_file):
        return self._create_example(
            self._read_data(data_file), "train"
        )

    def get_dev_examples(self, data_file):
        return self._create_example(
            self._read_data(data_file), "dev"
        )

    def get_test_examples(self, data_file):
        return self._create_example(
            self._read_data(data_file), "test"
        )

    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        return ner_label.pers_info_ner_labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[0])
            labels = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :return: feature
    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for j, _ in enumerate(token):
            if j == 0:
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
        input_mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature


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


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def convert_examples_to_features(examples, label_list, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, FLAGS.max_seq_length, tokenizer)
        features.append(feature)
    return features


def get_input_features(examples, label_list, tokenizer, reload, set_type):
    if reload:
        with open(os.path.join(FLAGS.data_dir, '%s_features.pkl' % set_type), 'rb') as f:
            features = pickle.load(f)
            print('Reloaded %d tokenized example' % len(features))
    else:
        features = convert_examples_to_features(examples, label_list, tokenizer)
        with open(os.path.join(FLAGS.data_dir, '%s_features.pkl' % set_type), 'wb') as f:
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
        batch_label_id = [item.label_ids for item in batch_data]
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
        self.tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=False)
        self.data_processor = NerProcessor()
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
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.loss, self.logits, self.output = self.create_model()

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

    def create_model(self, use_one_hot_embeddings=True):
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.ids_placeholder,
            input_mask=self.mask_placeholder,
            token_type_ids=self.segment_placeholder,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        output_layer = model.get_sequence_output()
        # output_layer shape is
        if self.is_training:
            output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
        logits = hidden2tag(output_layer, self.num_labels)
        # TODO test shape
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, self.num_labels])
        if FLAGS.crf:
            mask2len = tf.reduce_sum(self.mask_placeholder, axis=1)
            loss, trans = crf_loss(logits, self.labels_placeholder, self.mask_placeholder, self.num_labels, mask2len)
            predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
            return loss, logits, predict
        else:
            loss, predict = softmax_layer(logits, self.labels_placeholder, self.num_labels, self.mask_placeholder)
            return loss, logits, predict

    def train(self, data_file=FLAGS.training_data, reload=False, save_path=FLAGS.output_dir):
        examples = self.data_processor.get_train_examples(data_file)
        input_features = get_input_features(examples, self.labels, self.tokenizer, reload, set_type='train')
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
            saver = tf.train.Saver(max_to_keep=0, save_relative_paths=True)
            save_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H_%M') + '_' +
                                     str(FLAGS.num_train_epochs) + 'epoch')
            for epoch in range(FLAGS.num_train_epochs):
                batch, correct_count, label_count, predict_count = 0, 0, 0, 0
                for ids, mask, segment, labels in generate_batch(input_features, batch_size=FLAGS.train_batch_size):
                    train_dict = {self.ids_placeholder: ids,
                                  self.mask_placeholder: mask,
                                  self.segment_placeholder: segment,
                                  self.labels_placeholder: labels}
                    _, train_loss, output = self.sess.run([train_op, self.loss, self.output], feed_dict=train_dict)

                    # 统计准召
                    for i in range(FLAGS.train_batch_size):
                        out, label = [], []
                        for idx in output[i]:
                            if idx == 0:
                                break
                            if idx != 2:
                                out.append(self.labels[idx])
                        for idx in labels[i]:
                            if idx == 0:
                                break
                            if idx != 2:
                                label.append(self.labels[idx])
                        corr, label, pred = self.stat(out[1:], label[1:])
                        correct_count += corr
                        label_count += label
                        predict_count += pred
                    precision = round(correct_count / predict_count, 3) if predict_count != 0 else 0.0
                    recall = round(correct_count / label_count, 3) if label_count != 0 else 0.0
                    F1 = round(2 * precision * recall / (precision + recall), 3) if (
                            precision + recall != 0) else 0.0

                    if batch % 10 == 0:
                        print('Epoch: %d, batch: %d, training loss: %s, precision: %s, recall: %s, F1: %s'
                              % (epoch, batch, train_loss, precision, recall, F1))
                    if batch % FLAGS.save_checkpoints_steps == 0:
                        saver.save(self.sess, os.path.join(save_path, 'model.ckpt.' + str(epoch) + '.' + str(batch)))
                    batch += 1

    def evaluate(self, data_file=FLAGS.test_data, reload=False):
        examples = self.data_processor.get_test_examples(data_file)
        input_features = get_input_features(examples, self.labels, self.tokenizer, reload, set_type='eval')
        with self.graph.as_default():
            self.sess.run(tf.initialize_all_variables())
        correct_count, label_count, predict_count = 0, 0, 0
        for ids, mask, segment, labels in generate_batch(input_features, batch_size=FLAGS.eval_batch_size):
            eval_dict = {self.ids_placeholder: ids,
                         self.mask_placeholder: mask,
                         self.segment_placeholder: segment,
                         self.labels_placeholder: labels}
            output = self.sess.run(self.output, feed_dict=eval_dict)
            for i in range(FLAGS.eval_batch_size):
                out, label = [], []
                for idx in output[i]:
                    if idx == 0:
                        break
                    if idx != 2:
                        out.append(self.labels[idx])
                for idx in labels[i]:
                    if idx == 0:
                        break
                    if idx != 2:
                        label.append(self.labels[idx])
                corr, label, pred = self.stat(out[1:], label[1:])
                correct_count += corr
                label_count += label
                predict_count += pred

        precision = round(correct_count / predict_count, 3) if predict_count != 0 else 0.0
        recall = round(correct_count / label_count, 3) if label_count != 0 else 0.0
        F1 = round(2 * precision * recall / (precision + recall), 3) if (
                    predict_count + recall != 0) else 0.0
        print('Total precision: %s， total recall: %s, total F1: %s' % (precision, recall, F1))

    @staticmethod
    def stat(predict, label):
        def find_entity(symbols):
            entity, in_flag, start_idx = [], False, 0
            for i, ent in enumerate(symbols):
                if ent.startswith('B'):
                    if in_flag:
                        entity.append([start_idx, i, symbols[i - 1][2:]])
                        start_idx = i
                    else:
                        in_flag = True
                        start_idx = i
                elif ent.startswith('I'):
                    if ent[2:] != symbols[i - 1][2:] and in_flag:
                        entity.append([start_idx, i, symbols[i - 1][2:]])
                        in_flag = False
                elif ent.startswith('O'):
                    if in_flag:
                        entity.append([start_idx, i, symbols[i - 1][2:]])
                        in_flag = False
            return entity

        pred_ent = find_entity(predict)
        label_ent = find_entity(label)
        pred_count = len(pred_ent)
        label_count = len(label_ent)
        corr_count = 0

        for item in pred_ent:
            for ent in label[item[0]:item[1]]:
                if item[2] == ent[2:]:
                    corr_count += 1
                    break
        return corr_count, label_count, pred_count

    def predict(self, segment=''):
        example = self.data_processor._create_example([[segment, 'O '*len(segment.split(' '))]], set_type='predict')
        feature = convert_single_example(0, example[0], self.labels, FLAGS.max_seq_length, self.tokenizer)
        predict_dict = {self.ids_placeholder: [feature.input_ids],
                        self.mask_placeholder: [feature.input_mask],
                        self.segment_placeholder: [feature.segment_ids],
                        self.labels_placeholder: [feature.label_ids]}
        result = self.sess.run(self.output, feed_dict=predict_dict)
        res = []
        if FLAGS.crf:
            predictions = []
            for m, pred in enumerate(result):
                predictions.extend(pred)
            for i, prediction in enumerate(predictions):
                predict = self.labels[prediction]
                if predict != 'X':
                    res.append(predict)
        else:
            for i, prediction in enumerate(result):
                predict = self.labels[prediction]
                if predict != 'X':
                    res.append(predict)
        return res


def main(_):
    segment = "What is a Flash cookie? Local storage objects, also known as Flash cookies, are similar in function " \
              "to browser cookies in that they store some information about you or your activities on our Websites. " \
              "We use Flash cookies in certain situations where we use Flash to provide some content such as video " \
              "clips or animation. The options within your browser may not prevent the setting of Flash cookies. " \
              "To manage Flash cookies please click here: " \
              "http://www.macromedia.com/support/documentation/en/flashplayer/help/settings_manager07.html. <br> <br>"
    # checkpoint = FLAGS.init_checkpoint
    # checkpoint = tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'conll2003'))
    # checkpoint = tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'attribute_model'))
    checkpoint = tf.train.latest_checkpoint(os.path.join(FLAGS.output_dir, '2020-07-20_00_4epoch'))
    # clf = BertNer(is_training=True, init_checkpoint=checkpoint)
    # clf.train(reload=False)
    # clf = BertNer(is_training=False, init_checkpoint=checkpoint)
    # output = clf.predict(segment)
    # print(output)
    clf = BertNer(is_training=False, init_checkpoint=checkpoint)
    clf.evaluate()


if __name__ == "__main__":
    tf.app.run()
