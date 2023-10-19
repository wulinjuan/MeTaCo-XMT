import csv
import pickle
import string
from random import sample

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer

import sys

sys.path.append("/data/home10b/wlj/code/Meta_MRC/src")
from Syntax_distance_model_ablation.Dependency_paring import task
from Syntax_distance_model_ablation.decorators import auto_init_args

punc = string.punctuation

MODEL_CLASSES = {
    'mbert': BertTokenizer,
    'xlm': XLMTokenizer,
    'xlmr': XLMRobertaTokenizer
}

MAX_LEN = 256

def read_annotated_file(path):
    originals = []
    translations = []
    z_means = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        if "QE" in path:
            data_name = 'WNT2020_QE'
            for row in reader:
                originals.append(row["original"])
                translations.append(row["translation"])
                z_means.append(float(row['mean']))
        else:
            data_name = "zh_dev"
            for row in reader:
                try:
                    z_means.append(float(row['score'].strip()))
                    originals.append(row["text_a"])
                    translations.append(row["text_b"])
                except ValueError:
                    print(row)
    return {'data_name': data_name, 'original': originals, 'translation': translations, 'z_mean': z_means}


class data_holder:
    @auto_init_args
    def __init__(self, train_data, dev_data, test_data, vocab=None):
        if vocab is not None:
            self.inv_vocab = {i: w for w, i in vocab.items()}

    
class data_processor_eng:
    @auto_init_args
    def __init__(self, dp_train, dp_dev, dp_test, experiment):
        self.expe = experiment
        self.dp_train = dp_train
        self.dp_dev = dp_dev
        self.dp_test = dp_test

    def process(self):
        vocab = self._build_pretrain_vocab(self.expe.config.ml_type)
        if self.expe.config.ml_type == "xlm":
            v = vocab.decoder
        else:
            v = vocab.vocab
        self.expe.log.info("vocab size: {}".format(len(v)))

        train_data = self._data_to_idx_dp(vocab, dp1=self.dp_train)  # sample(self.dp_train, 10000)
        dev_data = self._data_to_idx_dp(vocab, dp1=self.dp_dev)
        test_data = self._data_to_idx_dp(vocab, dp1=self.dp_test)

        data = data_holder(train_data=train_data, dev_data=dev_data, test_data=test_data, vocab=v)

        return data, vocab

    def _data_to_idx_dp(self, vocab, dp1):
        idx_pair1 = []
        mask_idx = []

        dp1_depth_list = []
        dp1_dictance_list = []
        depth_get = task.ParseDepthTask
        distance_get = task.ParseDistanceTask
        for d1 in dp1:
            d1_dep, sentence1 = depth_get.labels(d1)
            s1, d1_dep, mask = self.load_depth_tag(sentence1, d1_dep, vocab)
            dp1_depth_list.append(d1_dep)
            d1_dis = distance_get.labels(d1)
            d1_dis = self.load_distance_tag(sentence1, d1_dis, vocab)
            dp1_dictance_list.append(d1_dis)
            idx_pair1.append(s1)
            mask_idx.append(mask)

        all_input_ids = torch.tensor([f for f in idx_pair1], dtype=torch.long)
        all_mask_ids = torch.tensor([f for f in mask_idx], dtype=torch.long)
        all_depth_list = torch.tensor([f for f in dp1_depth_list], dtype=torch.long)
        all_dictance_list = torch.tensor([f for f in dp1_dictance_list], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_mask_ids, all_depth_list, all_dictance_list
        )

        return dataset

    def load_depth_tag(self, words, tags, vocab):
        assert len(words) == len(tags)
        x, y = [vocab.convert_tokens_to_ids("[CLS]")], [-1]
        mask = [1]
        for w, t in zip(words, tags):
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                t = [t] + [-1] * (len(tokens) - 1)
                x.extend(xx)
                mask.extend([1 for _ in range(len(xx))])
                y.extend(t)
            assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))

        x.append(vocab.convert_tokens_to_ids("[SEP]"))
        y.append(-1)
        mask.append(0)
        if len(x) >=MAX_LEN:
            print(len(x))
        while len(x) < MAX_LEN:
            x.append(vocab.convert_tokens_to_ids("[PAD]"))
            y.append(-1)
            mask.append(0)
        assert len(x) == len(y) == len(mask), "len(x)={}, len(y)={}, len(mask)={}".format(len(x), len(y), len(mask))
        return x, y, mask

    def load_distance_tag(self, words, tags, vocab):
        assert len(words) == len(tags)
        x, y = [vocab.convert_tokens_to_ids("[CLS]")], []
        # x, y = [], []
        for w, t in zip(words, tags):
            if y == []:
                y = [np.ones(MAX_LEN) * (-1)]
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)
                # s = t.shape
                t_ = np.ones(MAX_LEN) * (-1)
                t_[1:t.shape[0]+1] = t
                t = [t_] + [np.ones(MAX_LEN) * (-1)] * (len(tokens) - 1)
                x.extend(xx)
                y.extend(t)
            assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))

        x.append(vocab.convert_tokens_to_ids("[SEP]"))
        y.extend([np.ones(MAX_LEN) * (-1)])
        while len(x) < MAX_LEN:
            x.append(vocab.convert_tokens_to_ids("[PAD]"))
            y.extend([np.ones(MAX_LEN) * (-1)])
        assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))
        return y

    def _build_pretrain_vocab(self, lm_type):
        Token = MODEL_CLASSES[lm_type]
        vocab = Token.from_pretrained(self.expe.config.ml_token)
        return vocab
    

class data_processor_pud:
    @auto_init_args
    def __init__(self, dp_eng, dp_pos, dp_neg, experiment):
        self.expe = experiment
        self.dp_eng = dp_eng
        self.dp_pos = dp_pos
        self.dp_neg = dp_neg

    def process(self):
        vocab = self._build_pretrain_vocab(self.expe.config.ml_type)
        # self.expe.log.info("vocab size: {}".format(len(v)))

        train_data, dev_data, test_data = self._data_to_idx_dp(vocab, dp1=self.dp_eng, dp2=self.dp_pos, dp3=self.dp_neg)  # sample(self.dp_train, 10000)

        data = data_holder(train_data=train_data, dev_data=dev_data, test_data=test_data)

        return data

    def _data_to_idx_dp(self, vocab, dp1, dp2, dp3):
        idx_pair1 = []
        mask_idx1 = []

        dp1_depth_list = []
        dp1_dictance_list = []

        idx_pair2 = []
        mask_idx2 = []

        dp2_depth_list = []
        dp2_dictance_list = []

        idx_pair3 = []
        mask_idx3 = []

        dp3_depth_list = []
        dp3_dictance_list = []
        depth_get = task.ParseDepthTask
        distance_get = task.ParseDistanceTask
        for d1 in dp1:
            d1_dep, sentence1 = depth_get.labels(d1)
            s1, d1_dep, mask = self.load_depth_tag(sentence1, d1_dep, vocab)
            dp1_depth_list.append(d1_dep)
            d1_dis = distance_get.labels(d1)
            d1_dis = self.load_distance_tag(sentence1, d1_dis, vocab)
            dp1_dictance_list.append(d1_dis)
            idx_pair1.append(s1)
            mask_idx1.append(mask)

        for d1 in dp2:
            d1_dep, sentence1 = depth_get.labels(d1)
            s1, d1_dep, mask = self.load_depth_tag(sentence1, d1_dep, vocab)
            dp2_depth_list.append(d1_dep)
            d1_dis = distance_get.labels(d1)
            d1_dis = self.load_distance_tag(sentence1, d1_dis, vocab)
            dp2_dictance_list.append(d1_dis)
            idx_pair2.append(s1)
            mask_idx2.append(mask)
        
        for d1 in dp3:
            d1_dep, sentence1 = depth_get.labels(d1)
            s1, d1_dep, mask = self.load_depth_tag(sentence1, d1_dep, vocab)
            dp3_depth_list.append(d1_dep)
            d1_dis = distance_get.labels(d1)
            d1_dis = self.load_distance_tag(sentence1, d1_dis, vocab)
            dp3_dictance_list.append(d1_dis)
            idx_pair3.append(s1)
            mask_idx3.append(mask)
        assert len(idx_pair3) == len(idx_pair2) == len(idx_pair1)
        
        train_num = sample([i for i in range(len(idx_pair3))], 6000)
        dev_num = sample([i for i in range(len(idx_pair3)) if i not in train_num], 500)
        all_input_ids1 = torch.tensor([f for i, f in enumerate(idx_pair1) if i in train_num], dtype=torch.long)
        all_mask_ids1 = torch.tensor([f for i, f in enumerate(mask_idx1) if i in train_num], dtype=torch.long)
        all_depth_list1 = torch.tensor([f for i, f in enumerate(dp1_depth_list) if i in train_num], dtype=torch.long)
        all_dictance_list1 = torch.tensor([f for i, f in enumerate(dp1_dictance_list) if i in train_num], dtype=torch.long)

        all_input_ids2 = torch.tensor([f for i, f in enumerate(idx_pair2) if i in train_num], dtype=torch.long)
        all_mask_ids2 = torch.tensor([f for i, f in enumerate(mask_idx2) if i in train_num], dtype=torch.long)
        all_depth_list2 = torch.tensor([f for i, f in enumerate(dp2_depth_list) if i in train_num], dtype=torch.long)
        all_dictance_list2 = torch.tensor([f for i, f in enumerate(dp2_dictance_list) if i in train_num], dtype=torch.long)

        all_input_ids3 = torch.tensor([f for i, f in enumerate(idx_pair3) if i in train_num], dtype=torch.long)
        all_mask_ids3 = torch.tensor([f for i, f in enumerate(mask_idx3) if i in train_num], dtype=torch.long)
        all_depth_list3 = torch.tensor([f for i, f in enumerate(dp3_depth_list) if i in train_num], dtype=torch.long)
        all_dictance_list3 = torch.tensor([f for i, f in enumerate(dp3_dictance_list) if i in train_num], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids1, all_mask_ids1, all_depth_list1, all_dictance_list1,
            all_input_ids2, all_mask_ids2, all_depth_list2, all_dictance_list2,
            all_input_ids3, all_mask_ids3, all_depth_list3, all_dictance_list3,
        )

        all_t__input_ids1 = torch.tensor([f for i, f in enumerate(idx_pair1) if i in dev_num], dtype=torch.long)
        all_t__mask_ids1 = torch.tensor([f for i, f in enumerate(mask_idx1) if i in dev_num], dtype=torch.long)
        all_t__depth_list1 = torch.tensor([f for i, f in enumerate(dp1_depth_list) if i in dev_num], dtype=torch.long)
        all_t__dictance_list1 = torch.tensor([f for i, f in enumerate(dp1_dictance_list) if i in dev_num],
                                          dtype=torch.long)

        all_t__input_ids2 = torch.tensor([f for i, f in enumerate(idx_pair2) if i in dev_num], dtype=torch.long)
        all_t__mask_ids2 = torch.tensor([f for i, f in enumerate(mask_idx2) if i in dev_num], dtype=torch.long)
        all_t__depth_list2 = torch.tensor([f for i, f in enumerate(dp2_depth_list) if i in dev_num], dtype=torch.long)
        all_t__dictance_list2 = torch.tensor([f for i, f in enumerate(dp2_dictance_list) if i in dev_num],
                                          dtype=torch.long)

        all_t__input_ids3 = torch.tensor([f for i, f in enumerate(idx_pair3) if i in dev_num], dtype=torch.long)
        all_t__mask_ids3 = torch.tensor([f for i, f in enumerate(mask_idx3) if i in dev_num], dtype=torch.long)
        all_t__depth_list3 = torch.tensor([f for i, f in enumerate(dp3_depth_list) if i in dev_num], dtype=torch.long)
        all_t__dictance_list3 = torch.tensor([f for i, f in enumerate(dp3_dictance_list) if i in dev_num],
                                          dtype=torch.long)

        dataset_test = TensorDataset(
            all_t__input_ids1, all_t__mask_ids1, all_t__depth_list1, all_t__dictance_list1,
            all_t__input_ids2, all_t__mask_ids2, all_t__depth_list2, all_t__dictance_list2,
            all_t__input_ids3, all_t__mask_ids3, all_t__depth_list3, all_t__dictance_list3,
        )

        all_t__input_ids1 = torch.tensor([f for i, f in enumerate(idx_pair1) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__mask_ids1 = torch.tensor([f for i, f in enumerate(mask_idx1) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__depth_list1 = torch.tensor([f for i, f in enumerate(dp1_depth_list) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__dictance_list1 = torch.tensor([f for i, f in enumerate(dp1_dictance_list) if i not in dev_num and i not in train_num],
                                             dtype=torch.long)

        all_t__input_ids2 = torch.tensor([f for i, f in enumerate(idx_pair2) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__mask_ids2 = torch.tensor([f for i, f in enumerate(mask_idx2) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__depth_list2 = torch.tensor([f for i, f in enumerate(dp2_depth_list) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__dictance_list2 = torch.tensor([f for i, f in enumerate(dp2_dictance_list) if i not in dev_num and i not in train_num],
                                             dtype=torch.long)

        all_t__input_ids3 = torch.tensor([f for i, f in enumerate(idx_pair3) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__mask_ids3 = torch.tensor([f for i, f in enumerate(mask_idx3) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__depth_list3 = torch.tensor([f for i, f in enumerate(dp3_depth_list) if i not in dev_num and i not in train_num], dtype=torch.long)
        all_t__dictance_list3 = torch.tensor([f for i, f in enumerate(dp3_dictance_list) if i not in dev_num and i not in train_num],
                                             dtype=torch.long)

        dataset_dev = TensorDataset(
            all_t__input_ids1, all_t__mask_ids1, all_t__depth_list1, all_t__dictance_list1,
            all_t__input_ids2, all_t__mask_ids2, all_t__depth_list2, all_t__dictance_list2,
            all_t__input_ids3, all_t__mask_ids3, all_t__depth_list3, all_t__dictance_list3,
        )

        return dataset, dataset_dev, dataset_test

    def load_depth_tag(self, words, tags, vocab):
        assert len(words) == len(tags)
        x, y = [vocab.convert_tokens_to_ids("[CLS]")], [-1]
        mask = [1]
        for w, t in zip(words, tags):
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                t = [t] + [-1] * (len(tokens) - 1)
                x.extend(xx)
                mask.extend([1 for _ in range(len(xx))])
                y.extend(t)
            assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))

        x.append(vocab.convert_tokens_to_ids("[SEP]"))
        y.append(-1)
        mask.append(0)
        if len(x) >=MAX_LEN:
            print(len(x))
        while len(x) < MAX_LEN:
            x.append(vocab.convert_tokens_to_ids("[PAD]"))
            y.append(-1)
            mask.append(0)
        assert len(x) == len(y) == len(mask), "len(x)={}, len(y)={}, len(mask)={}".format(len(x), len(y), len(mask))
        return x, y, mask

    def load_distance_tag(self, words, tags, vocab):
        assert len(words) == len(tags)
        x, y = [vocab.convert_tokens_to_ids("[CLS]")], []
        # x, y = [], []
        for w, t in zip(words, tags):
            if y == []:
                y = [np.ones(MAX_LEN) * (-1)]
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)
                # s = t.shape
                t_ = np.ones(MAX_LEN) * (-1)
                t_[1:t.shape[0]+1] = t
                t = [t_] + [np.ones(MAX_LEN) * (-1)] * (len(tokens) - 1)
                x.extend(xx)
                y.extend(t)
            assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))

        x.append(vocab.convert_tokens_to_ids("[SEP]"))
        y.extend([np.ones(MAX_LEN) * (-1)])
        while len(x) < MAX_LEN:
            x.append(vocab.convert_tokens_to_ids("[PAD]"))
            y.extend([np.ones(MAX_LEN) * (-1)])
        assert len(x) == len(y), "len(x)={}, len(y)={}".format(len(x), len(y))
        return y

    def _build_pretrain_vocab(self, lm_type):
        Token = MODEL_CLASSES[lm_type]
        vocab = Token.from_pretrained(self.expe.config.ml_token)
        return vocab