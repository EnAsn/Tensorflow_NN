__version__ = "1.0.0"

import numpy as np
from os import path
import json
import datetime

import jieba.posseg as pseg
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from sklearn.metrics import precision_recall_fscore_support as score


class BasicLoader:
    def __init__(self, config_path):
        self.config_file_path = config_path

    @staticmethod
    def get_path(relative_path):
        return path.join(path.dirname(__file__), relative_path)

    @staticmethod
    def convert_boolean(_value):
        return True if _value == 'True' else False

    @staticmethod
    def convert_integer(_value):
        return int(_value)

    @staticmethod
    def convert_float(_value):
        return float(_value)

    @staticmethod
    def convert_string(_value):
        return _value

    def convert_path(self, _value):
        return self.get_path(_value)

    def load_flags(self, target_flags=list()):
        flags = dict()
        convert_func = {
            "boolean": self.convert_boolean,
            "integer": self.convert_integer,
            "float": self.convert_float,
            "path": self.convert_path
        }
        with open(self.config_file_path) as config_file:
            configs = json.load(config_file)
            for data_type in convert_func:
                for config in configs[data_type]:
                    if target_flags:
                        if config['attr'] in target_flags:
                            flags[config['attr']] = convert_func[data_type](config['value'])
                    else:
                        flags[config['attr']] = convert_func[data_type](config['value'])
            return flags

    @staticmethod
    def select_dict_by_key(_dict, _keys=list()):
        config = dict()
        for attr in _keys:
            config[attr] = _dict[attr]
        return config

    def load_model_config(self, all_configs, is_train):
        keys_for_model = ['num_epochs', 'lr', 'batch_size',
                          'vocabulary_size', 'num_pos', 'num_chosen_decision', 'num_classes',
                          'embedding_size', 'pos_emb_size', 'chosen_emb_size',
                          'num_steps', 'num_GRU_layers', 'num_units',
                          'dropout_rate', 'pre_trained_embedding']
        config = self.select_dict_by_key(all_configs, _keys=keys_for_model)
        if not is_train:
            config['batch_size'] = 1
        return config

    @staticmethod
    def sentence2id(sentence, _w2i, _p2i, fix_len=100, given_chosen_list=list(), c_dict=None, seg_method=pseg.cut):
        seg_result = seg_method(sentence)
        counter = 0
        sent_current_index = 0
        seg2id = list()
        pos2id = list()
        chosen2id = list()
        word_list = list()
        for _word, _pos in seg_result:
            word_list.append((_word, _pos))
            counter += 1
            if counter > fix_len:
                break
            if _word == ' ':
                emb_id = _w2i['SPACE']
            else:
                emb_id = _w2i[_word] if _word in _w2i.keys() else _w2i['UNK']
            seg2id.append(emb_id)
            pos_id = _p2i[_pos]
            pos2id.append(pos_id)
            if given_chosen_list:
                chosen_flag = 0
                for i in range(len(_word)):
                    if given_chosen_list[sent_current_index + i]:
                        chosen_flag = 1
                        break
                chosen2id.append(chosen_flag)
            else:
                chosen2id.append(c_dict[_pos])

            sent_current_index += len(_word)

        if counter < 100:
            for i in range(counter, 100):
                seg2id.append(_w2i['BLANK'])
                pos2id.append(_p2i['aba'])
                chosen2id.append(0)
        # print(word_list)
        return seg2id, pos2id, chosen2id

    def tokenizer(self, _input, _w2i, _p2i, chosen_list, data_type='test', fix_len=100, c_dict=None, label=None):
        if data_type.startswith('test'):
            _seg2id, _pos2id, _chosen2id = self.sentence2id(_input, _w2i, _p2i,
                                                            given_chosen_list=chosen_list,
                                                            fix_len=fix_len, c_dict=c_dict)
            return _seg2id, _pos2id, _chosen2id
        else:
            one_hot_label = [0] * 2
            one_hot_label[label] = 1
            _seg2id, _pos2id, _chosen2id = self.sentence2id(_input, _w2i, _p2i, given_chosen_list=chosen_list,
                                                            fix_len=fix_len)
            return _seg2id, _pos2id, _chosen2id, one_hot_label

    @staticmethod
    def column_sentence2row(file_path):
        char_catcher = ''
        pos_cather = list()
        data_set = list()
        with open(file_path, encoding='utf-8') as f:
            for _row in f:
                row = _row.strip()
                if len(row) != 1:
                    character, if_chosen = row.split(' ')
                    char_catcher += character
                    pos_cather.append(int(if_chosen))
                else:
                    if char_catcher:
                        new_sentence = char_catcher
                        data_set.append({
                            'sentence': new_sentence,
                            'chosen_list': pos_cather,
                            'isPos': int(row)
                        })
                        char_catcher = ''
                        pos_cather = list()
                    else:
                        pass
        return data_set

    def load_data(self, file_path, _word2id, _pos_dict, data_type='train', fix_len=100, _c_dict=None):
        sentence_list = list()
        pos_list = list()
        chosen_list = list()
        label_list = list()
        if data_type == 'train' or data_type == 'evaluate':
            f = self.column_sentence2row(file_path)
            for _item in f:
                sentence_token, pos_token, chosen_token, label_token = self.tokenizer(_item['sentence'],
                                                                                      _word2id, _pos_dict,
                                                                                      chosen_list=_item['chosen_list'],
                                                                                      label=_item['isPos'],
                                                                                      data_type=data_type,
                                                                                      fix_len=fix_len)
                sentence_list.append(sentence_token)
                pos_list.append(pos_token)
                chosen_list.append(chosen_token)
                label_list.append(label_token)
            data_set = [sentence_list, pos_list, chosen_list, label_list]
        else:
            with open(file_path, encoding='utf-8') as f:
                for row in f:
                    sentence_token, pos_token, chosen_token, _ = self.tokenizer(row, _word2id, _pos_dict,
                                                                                chosen_list=[], c_dict=_c_dict,
                                                                                data_type=data_type, fix_len=fix_len)
                    sentence_list.append(sentence_token)
                    pos_list.append(pos_token)
                    chosen_list.append(chosen_token)
                data_set = [sentence_list, pos_list, chosen_list]
        return data_set

    @staticmethod
    def load_pos_types(pos_emb_path):
        pos_to_id = dict()
        with open(pos_emb_path, encoding='utf-8') as f:
            for row in f:
                if row:
                    pos, pid = row.strip().split()
                    pos_to_id[pos] = int(pid)
        return pos_to_id

    @staticmethod
    def load_chosen_pos(chosen_pos_path):
        if_chosen = dict()
        with open(chosen_pos_path, encoding='utf-8') as f:
            for row in f:
                if row:
                    pos, ifc = row.strip().split()
                    if_chosen[pos] = int(ifc)
        return if_chosen

    @staticmethod
    def load_word_embedding(embedding_path):
        word_to_id = dict()
        emb_lookup_table = list()
        with open(embedding_path, encoding='utf-8') as f:
            row = f.readline()
            size, dim = [int(x) for x in row.strip().split()]
            counter = 0
            for row in f:
                if row:
                    content = row.strip().split()
                    word_to_id[content[0]] = counter
                    emb_lookup_table.append(content[1:])
                    counter += 1

        # add UNKNOWN, SPACE, BLANK
        word_to_id['UNK'] = counter
        word_to_id['SPACE'] = counter + 1
        word_to_id['BLANK'] = counter + 2
        for v in range(3):
            emb_lookup_table.append(np.random.normal(size=dim, loc=0, scale=0.05))
        emb_lookup_table = np.array(emb_lookup_table, dtype=np.float32)

        print('Finish Loading Word Embedding Lookup Table')
        return word_to_id, emb_lookup_table

    @staticmethod
    def load_label(label_path):
        label_to_id = dict()
        with open(label_path, encoding='utf-8') as f:
            for row in f:
                if row:
                    desc, lid = row.strip().split()
                    label_to_id[lid] = {
                        "desc": desc,
                        "id": int(lid)
                    }
        return label_to_id


class BasicNNModel:
    def __init__(self, config, is_train):
        self.config = config
        self.num_classes = config['num_classes']
        self.vocabulary_size = config['vocabulary_size']
        self.embedding_size = config['embedding_size']

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.best_train_f1 = tf.Variable(0.0, trainable=False, name="best_train_f1")
        self.best_evaluate_f1 = tf.Variable(0.0, trainable=False, name="best_evaluate_f1")
        self.best_test_f1 = tf.Variable(0.0, trainable=False, name="best_test_f1")

        # Params for embedding layer
        self.word_embedding = None
        self.pos_embedding = None
        self.num_pos = config['num_pos']
        self.pos_emb_size = config['pos_emb_size']
        self.chosen_embedding = None
        self.num_chosen_decision = config['num_chosen_decision']
        self.chosen_emb_size = config['chosen_emb_size']

        # Params for GRU layer
        self.num_layers = config['num_GRU_layers']
        self.num_units = config['num_units']
        self.dropout_rate = config['dropout_rate']
        self.lr = config['lr']

        # Variables
        self.initializer = initializers.xavier_initializer()
        self.sents = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sents')
        self.poses = tf.placeholder(dtype=tf.int32, shape=[None, None], name='poses')
        self.chosen_pos = tf.placeholder(dtype=tf.int32, shape=[None, None], name='chosen_pos')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='label')

        self.batch_size = tf.shape(self.sents)[0]
        self.num_steps = tf.shape(self.sents)[1]

        self.y_truth = list()
        self.predictions = list()

        # Model Structure
        embedding_output_fw, embedding_output_bw = self.bi_embedding_layer(self.sents, self.poses, self.chosen_pos)
        bi_gru_output = self.bi_gru_layer(embedding_output_fw, embedding_output_bw, is_train)
        class_distributions = self.hidden_layer(bi_gru_output)
        self.loss, self.prob, self.predictions = self.output_layer(class_distributions)

        if is_train:
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables())

    def bi_embedding_layer(self, input_words, input_pos, input_chosen):
        """
        Embedding layer

        """
        self.word_embedding = tf.get_variable(
            'word_embedding', shape=[self.vocabulary_size, self.embedding_size],
            initializer=self.initializer)
        self.pos_embedding = tf.get_variable('pos_embedding', [self.num_pos, self.pos_emb_size])
        self.chosen_embedding = tf.get_variable('chosen_embedding', [self.num_chosen_decision, self.chosen_emb_size])

        embedded_fw = tf.concat(
            axis=2,
            values=[
                tf.nn.embedding_lookup(self.word_embedding, input_words),
                tf.nn.embedding_lookup(self.pos_embedding, input_pos),
                tf.nn.embedding_lookup(self.chosen_embedding, input_chosen),
            ]
        )

        embedded_bw = tf.concat(
            axis=2,
            values=[
                tf.nn.embedding_lookup(self.word_embedding, tf.reverse(input_words, [1])),
                tf.nn.embedding_lookup(self.pos_embedding, tf.reverse(input_pos, [1])),
                tf.nn.embedding_lookup(self.chosen_embedding, tf.reverse(input_chosen, [1])),
            ]
        )

        return embedded_fw, embedded_bw

    def bi_gru_layer(self, gru_fw_inputs, gru_bw_inputs, is_train):
        """
        Bi-GRU layer

        Args:
            gru_fw_inputs: forward embedding
            gru_bw_inputs: backward embedding
            is_train: True for training mode, otherwise False

        Returns:
            Output of Bi-GRU layer

        """

        # Bi-GRU layer
        gru_cell_forward = tf.contrib.rnn.GRUCell(self.num_units, reuse=tf.AUTO_REUSE)
        gru_cell_backward = tf.contrib.rnn.GRUCell(self.num_units, reuse=tf.AUTO_REUSE)

        if is_train and self.dropout_rate < 1:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(
                gru_cell_forward, output_keep_prob=1 - self.dropout_rate)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(
                gru_cell_backward, output_keep_prob=1 - self.dropout_rate)

        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * self.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * self.num_layers)

        output_fw, state_fw = tf.nn.dynamic_rnn(cell_forward, gru_fw_inputs, dtype=tf.float32)
        output_bw, state_bw = tf.nn.dynamic_rnn(cell_backward, gru_bw_inputs, dtype=tf.float32)
        output_h = tf.add(output_fw, output_bw)
        return output_h[:, -1, :]

    def hidden_layer(self, gru_output):
        with tf.variable_scope("tanh_layer", initializer=self.initializer):
            w = tf.get_variable(name="weights", shape=[self.num_units, self.num_units], dtype=tf.float32)
            b = tf.get_variable(name="bias", shape=[self.num_units], dtype=tf.float32)
            x = tf.reshape(gru_output, shape=[-1, self.num_units])
            hidden_output = tf.tanh(tf.nn.xw_plus_b(x, w, b))

        with tf.variable_scope("linear_layer", initializer=self.initializer):
            w = tf.get_variable(name="weights", shape=[self.num_units, self.num_classes], dtype=tf.float32)
            b = tf.get_variable(name="bias", shape=[self.num_classes], dtype=tf.float32)
            score = tf.nn.xw_plus_b(hidden_output, w, b)
        return tf.reshape(score, shape=[-1, self.num_classes])

    def output_layer(self, class_distributions):
        prob = tf.nn.softmax(class_distributions)
        predictions = tf.argmax(prob, 1)
        log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=class_distributions, labels=self.labels)
        loss = tf.reduce_mean(log_likelihood)
        return loss, prob, predictions

    def run_step(self, session, feed_dict, is_train):
        if is_train:
            global_step, final_loss, _, predictions, prob = session.run(
                [self.global_step, self.loss, self.train_op, self.predictions, self.prob], feed_dict)
            return global_step, final_loss, predictions, prob
        else:
            prob, predictions = session.run(
                [self.prob, self.predictions], feed_dict)
            return prob, predictions

    def evaluate(self, session, feed_dict):
        prob, predictions = self.run_step(session, feed_dict, is_train=False)
        return prob, predictions


class BasicSolution:
    def __init__(self, loader, is_train=False, basic_model=None):
        self.loader_obj = loader
        self.flags = self.loader_obj.load_flags()
        self.model_configs = None

        self.session = None
        self.model = None
        self.BasicModel = basic_model

        self.word_embedding = None
        self.topic2id = None
        self.word2id = None
        self.label2id = None

        self.pos_dict = self.loader_obj.load_pos_types(self.flags['pos_dict'])
        self.word2id, self.word_embedding = self.loader_obj.load_word_embedding(self.flags['word_embedding_path'])
        self.if_chosen_pos = self.loader_obj.load_chosen_pos(self.flags['if_chosen_pos'])
        self.label2id = self.loader_obj.load_label(self.flags['labels_path'])

        if not is_train:
            self.build(is_train)

    def build(self, is_train):
        self.model_configs = self.loader_obj.load_model_config(self.flags, is_train)

        tf.reset_default_graph()
        self.session = tf.Session()
        self.model = self.BasicModel(self.model_configs, is_train)

        ckpt = tf.train.get_checkpoint_state(path.join(path.dirname(__file__), self.flags['ckpt_path']))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Restore Model From ckpt')
            self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print('Build New Model...')
            self.session.run(tf.global_variables_initializer())
            if self.model_configs["pre_trained_embedding"]:
                self.session.run(self.model.word_embedding.assign(self.word_embedding))

    def load_data_set(self, data_type):
        return self.loader_obj.load_data(self.flags['%s_data_path' % data_type],
                                         self.word2id, self.pos_dict, data_type=data_type, fix_len=self.flags['fixlen'])

    def make_feed_dict(self, sents, poses, chosens, labels):
        feed_dict = dict()
        feed_dict[self.model.sents] = np.array(sents)
        feed_dict[self.model.poses] = np.array(poses)
        feed_dict[self.model.chosen_pos] = np.array(chosens)
        feed_dict[self.model.labels] = np.array(labels)
        return feed_dict

    def make_batch(self, dataset):
        batches = list()
        sentence_set, pos_set, chosen_set, label_set = dataset
        print(len(sentence_set))
        for i in range(len(sentence_set) // self.flags['batch_size'] + 1):
            start = i * self.flags['batch_size']
            end = (i+1) * self.flags['batch_size']
            batches.append([sentence_set[start:end], pos_set[start:end], chosen_set[start:end], label_set[start:end]])
        return batches

    def train(self):
        self.build(is_train=True)
        training_data_set = self.load_data_set('train')
        print('Training Data Loaded!')
        batches = self.make_batch(training_data_set)
        for one_epoch in range(self.model_configs['num_epochs']):
            print('No. %s Epoch' % one_epoch)
            for sentences, poses, chosens, labels in batches:
                feed_dict = self.make_feed_dict(sentences, poses, chosens, labels)
                # print(feed_dict)
                step, batch_loss, predictions, prob = self.model.run_step(self.session, feed_dict, is_train=True)
                time_str = datetime.datetime.now().isoformat()
                if step % 10 == 0:
                    y_truth = []
                    for each_one_hot_label in labels:
                        y_truth.append(np.argmax(each_one_hot_label))
                    print("{}: step {}, batch loss {:g}".format(time_str, step, batch_loss))

            self.evaluate("train")
            best = self.evaluate("evaluate")
            print('End of One Epoch!')

            if best:
                print('Saving Model...')
                _path = self.model.saver.save(self.session, self.flags['ckpt_path'] + 'suojv')
                print('Model Has Saved to ' + _path)

        self.session.close()

    def evaluate(self, _type):
        _sentence, _pos, _chosen, _label = self.load_data_set(_type)
        feed_dict = self.make_feed_dict(_sentence, _pos, _chosen, _label)
        prob, predictions = self.model.evaluate(self.session, feed_dict)

        y_truth = []
        for each_one_hot_label in _label:
            y_truth.append(np.argmax(each_one_hot_label))

        if _type == "test":
            for x, y in zip(y_truth, predictions):
                print('Case: ', x, y)
        # print(y_truth)
        # print(predictions)
        precision, recall, fscore, support = score(y_truth, predictions, warn_for=())
        print('Precision:\t %s' % ['{0:0.2f}'.format(x) for x in precision])
        print('Recall:\t %s' % ['{0:0.2f}'.format(x) for x in recall])
        print('F-score:\t %s' % ['{0:0.2f}'.format(x) for x in fscore])
        print('Support:\t %s' % ['{}'.format(x) for x in support])

        print('Overall Score')
        precision, recall, fscore, support = score(y_truth, predictions, warn_for=(), average='weighted')
        print('Precision:\t %s' % '{0:0.4f}'.format(precision))
        print('Recall:\t %s' % '{0:0.4f}'.format(recall))
        print('F-score:\t %s' % '{0:0.4f}'.format(fscore))
        print('Support:\t %s' % '{}'.format(support))
        f1 = fscore

        best_f1 = getattr(self.model, 'best_%s_f1' % _type).eval(self.session)
        if f1 > best_f1:
            tf.assign(getattr(self.model, 'best_%s_f1' % _type), f1).eval(session=self.session)
            print("New Best {} f1 Score:{:>3f}".format('Validation', f1))
        return f1 > best_f1

    def predict(self, content):
        test_sentence, test_pos, test_chosen = self.loader_obj.tokenizer(content, self.word2id, self.pos_dict,
                                                                         chosen_list=[], c_dict=self.if_chosen_pos,
                                                                         data_type='test', fix_len=100)
        feed_dict = self.make_feed_dict([test_sentence], [test_pos], [test_chosen], [[0, 0]])
        # print(feed_dict)
        prob, predictions = self.model.evaluate(self.session, feed_dict)
        probability = np.max(prob)
        predict = {v['id']: v['desc'] for k, v in self.label2id.items()}[predictions[0]]
        return predict, probability


if __name__ == '__main__':
    print(123)
