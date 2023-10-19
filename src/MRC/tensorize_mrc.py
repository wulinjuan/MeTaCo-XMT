import os
import sys
from os.path import join
import pickle
import logging

import numpy as np

sys.path.append("../../src/")
from squad import SquadProcessor, squad_convert_examples_to_features, squad_convert_examples_to_meta_features
from MRC import util

logger = logging.getLogger(__name__)


class QaDataProcessor:
    def __init__(self, config):
        self.config = config

        self.max_seg_len = config['max_segment_len']
        self.doc_stride = config['doc_stride']
        self.max_query_len = config['max_query_len']

        self.tokenizer = None  # Lazy loading

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = util.get_bert_tokenizer(self.config)
        return self.tokenizer

    def _get_data(self, dataset_name, partition, lang, data_dir, data_file, cl_model=None,
                  model_path=None):
        cache_feature_path = self.get_cache_feature_path(dataset_name, partition, lang)
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, lang)
        if os.path.exists(cache_feature_path):
            with open(cache_feature_path, 'rb') as f:
                examples, features = pickle.load(f)
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            to_return = (examples, features, dataset)
            logger.info('Loaded features and dataset from cache')
        else:
            if partition == 'train':
                # 加载support候选集——英文训练集
                cache_feature_path = self.get_cache_feature_path(dataset_name, "train", 'en')
                if os.path.exists(cache_feature_path):
                    with open(cache_feature_path, 'rb') as f:
                        examples, train_features = pickle.load(f)
                else:
                    processor = SquadProcessor()
                    assert dataset_name in ['squad', 'tydiqa']
                    if dataset_name == 'squad' or dataset_name == 'mlqa':
                        data_dir = join(self.config['download_dir'], dataset_name)
                        data_file = f'train-v1.1.json'
                    else:
                        data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-train')
                        data_file = f'tydiqa.en.train.json'
                    examples = processor.get_train_examples(data_dir, data_file)
                    train_features = squad_convert_examples_to_features(examples, self.get_tokenizer(),
                                                                        max_seq_length=self.max_seg_len,
                                                                        doc_stride=self.doc_stride,
                                                                        max_query_length=self.max_query_len,
                                                                        config=self.config,
                                                                        is_training=True,
                                                                        model_path=model_path,
                                                                        get_meta_data=False,
                                                                        return_dataset=False)
                logger.info(f'Getting {dataset_name}-{partition}-{lang}; results will be cached')

                # 加载query
                processor = SquadProcessor()
                assert dataset_name in ['squad', 'tydiqa']
                if dataset_name == 'squad':
                    data_dir = join(self.config['download_dir'], dataset_name)
                    data_file = f'dev-v1.1.json'
                else:
                    data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-train')
                    data_file = f'tydiqa.{lang}.dev.txt.json'
                examples = processor.get_train_examples(data_dir, data_file)
                features, dataset = squad_convert_examples_to_features(examples, self.get_tokenizer(),
                                                                       max_seq_length=self.max_seg_len,
                                                                       doc_stride=self.doc_stride,
                                                                       max_query_length=self.max_query_len,
                                                                       config=self.config,
                                                                       is_training=(partition == 'train'),
                                                                       get_meta_data=(partition == 'train'),
                                                                       return_dataset=True,
                                                                       train_features=train_features,
                                                                       cl_model=cl_model,
                                                                       model_path=model_path)
            else:
                processor = SquadProcessor()
                examples = processor.get_train_examples(data_dir, data_file) if partition == 'train' else \
                    processor.get_dev_or_test_examples(data_dir, data_file)  # For train, only use first answer as gold
                features, dataset = squad_convert_examples_to_features(examples, self.get_tokenizer(),
                                                                       max_seq_length=self.max_seg_len,
                                                                       doc_stride=self.doc_stride,
                                                                       max_query_length=self.max_query_len,
                                                                       config=self.config,
                                                                       is_training=(partition == 'train'),
                                                                       get_meta_data=(partition == 'train'),
                                                                       return_dataset=True,
                                                                       train_features=None,
                                                                       cl_model=cl_model,
                                                                       model_path=model_path)
            cache_feature_path = self.get_cache_feature_path(dataset_name, partition, lang)
            with open(cache_feature_path, 'wb') as f:
                pickle.dump((examples, features), f, protocol=4)
            max_bytes = 2 ** 31 - 1
            bytes_out = pickle.dumps(dataset)
            n_bytes = sys.getsizeof(bytes_out)
            with open(cache_dataset_path, 'wb') as f:
                for idx in range(0, n_bytes, max_bytes):
                    f.write(bytes_out[idx:idx + max_bytes])
                # pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info('Saved features and dataset to cache')
            to_return = (examples, features, dataset)
        return to_return

    def _get_meta_data(self, dataset_name, partition, lang, data_query, data_cand, cl_model=None,
                       model_path=None):
        cache_feature_path = self.get_cache_feature_path(dataset_name, partition, lang)
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, lang)
        if os.path.exists(cache_dataset_path):
            with open(cache_feature_path, 'rb') as f:
                examples, features = pickle.load(f)
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            to_return = (examples, features, dataset)
            logger.info('Loaded features and dataset from cache')
        else:
            with open(data_cand, 'rb') as f:
                examples = pickle.load(f)
            if len(examples) == 2:
                train_features = examples[1]
            else:
                train_features = squad_convert_examples_to_features(examples, self.get_tokenizer(),
                                                                    max_seq_length=self.max_seg_len,
                                                                    doc_stride=self.doc_stride,
                                                                    max_query_length=self.max_query_len,
                                                                    config=self.config,
                                                                    is_training=True,
                                                                    get_meta_data=False,
                                                                    return_dataset=False,
                                                                    model_path=model_path,
                                                                    cand_features_file='../../data/meta_data/cand_features.bin')
            logger.info(f'Getting {dataset_name}-{partition}-{lang}; results will be cached')

            with open(data_query, 'rb') as f:
                examples = pickle.load(f)
            if cl_model is not None:
                cl = "cl"
            else:
                cl = ""
            features, dataset = squad_convert_examples_to_features(examples, self.get_tokenizer(),
                                                                   max_seq_length=self.max_seg_len,
                                                                   doc_stride=self.doc_stride,
                                                                   max_query_length=self.max_query_len,
                                                                   config=self.config,
                                                                   is_training=(partition == 'train'),
                                                                   get_meta_data=(partition != 'dev.txt'),
                                                                   return_dataset=True,
                                                                   train_features=train_features,
                                                                   cl_model=cl_model,
                                                                   model_path=model_path,
                                                                   cand_features_file=f"../../data/cache/meta_data/{self.config.model_type}/train.en.{cl}.{self.config.max_segment_len}.{self.config.meta_example_num}.bin")
            """cache_feature_path = self.get_cache_feature_path(dataset_name, partition, lang)
            with open(cache_feature_path, 'wb') as f:
                pickle.dump((examples, features), f, protocol=4)
            max_bytes = 2 ** 31 - 1
            bytes_out = pickle.dumps(dataset)
            n_bytes = sys.getsizeof(bytes_out)
            with open(cache_dataset_path, 'wb') as f:
                for idx in range(0, n_bytes, max_bytes):
                    f.write(bytes_out[idx:idx + max_bytes])
                # pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info('Saved features and dataset to cache')"""
            to_return = (examples, features, dataset)
        return to_return

    def get_meta(self, dataset_name, partition, lang='en', only_dataset=False, cl_model=None, model_path=None):
        # 加载support候选集——英文训练集
        cache_feature_path = self.get_cache_feature_path(dataset_name, "train", 'en')
        if os.path.exists(cache_feature_path):
            with open(cache_feature_path, 'rb') as f:
                train_features = pickle.load(f)
        else:
            processor = SquadProcessor()
            assert dataset_name in ['squad', 'tydiqa', 'mlqa']
            if dataset_name == 'squad' or dataset_name == 'mlqa':
                data_dir = join(self.config['download_dir'], 'squad')
                data_file = f'train-v1.1.json'
            #elif dataset_name == 'mlqa':
            #    data_dir = join(self.config['download_dir'], dataset_name, f'MLQA_V1/dev.txt')
            #    data_file = f'dev.txt-context-en-question-en.json'
            else:
                data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-train')
                data_file = f'tydiqa.en.train.json'
            examples = processor.get_train_examples(data_dir, data_file)
            train_features = squad_convert_examples_to_features(examples, self.get_tokenizer(),
                                                                max_seq_length=self.max_seg_len,
                                                                doc_stride=self.doc_stride,
                                                                max_query_length=self.max_query_len,
                                                                config=self.config,
                                                                is_training=True,
                                                                model_path=model_path,
                                                                get_meta_data=False,
                                                                return_dataset=False)
            with open(cache_feature_path, 'wb') as f:
                pickle.dump(train_features, file=f)
        logger.info(f'Getting {dataset_name}-{partition}-{lang}; results will be cached')

        # 加载query
        if cl_model is not None:
            cache_feature_path = self.get_cache_feature_path_norm(dataset_name, "train", lang)
        else:
            cache_feature_path = None
        """if os.path.exists(cache_feature_path):
            with open(cache_feature_path, 'rb') as f:
                examples, qry_features, dataset = pickle.load(f)
        else:"""
        processor = SquadProcessor()
        assert dataset_name in ['squad', 'tydiqa', 'mlqa']
        if dataset_name == 'squad':
            data_dir = join(self.config['download_dir'], dataset_name, f'MLQA_V1/dev.txt')
            data_file = f'dev.txt-context-{lang}-question-{lang}.json'
        elif dataset_name == 'mlqa':
            data_dir = join(self.config['download_dir'], dataset_name, 'MLQA_V1/dev.txt/')
            data_file = f'dev.txt-context-{lang}-question-{lang}.json'
        else:
            if partition == 'train':
                data_p = 'train'
            else:
                data_p = 'dev.txt'
            data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-{data_p}')
            data_file = f'tydiqa.{lang}.dev.json'
        examples = processor.get_train_examples(data_dir, data_file)
        spt_features, qry_features, dataset = squad_convert_examples_to_meta_features(examples, self.get_tokenizer(),
                                                                                      max_seq_length=self.max_seg_len,
                                                                                      doc_stride=self.doc_stride,
                                                                                      max_query_length=self.max_query_len,
                                                                                      config=self.config,
                                                                                      is_training=(
                                                                                                  partition == 'train'),
                                                                                      get_meta_data=True,
                                                                                      return_dataset=True,
                                                                                      train_features=train_features,
                                                                                      cl_model=cl_model,
                                                                                      model_path=model_path,
                                                                                      query_feature_file=cache_feature_path)
        if partition == 'meta_test':
            examples = processor.get_dev_or_test_examples(data_dir, data_file)

        # with open(cache_feature_path, 'wb') as f:
        #     pickle.dump((examples, qry_features, dataset), f, protocol=4)

        return dataset if only_dataset else (examples, qry_features, dataset)

    def get_source(self, dataset_name, partition, lang='en', only_dataset=False, cl_model=None, model_path=None):
        assert dataset_name in ['squad', 'tydiqa', 'mlqa']
        if dataset_name == 'squad':
            data_dir = join(self.config['download_dir'], dataset_name)
            data_file = f'{partition}-v1.1.json'
        elif dataset_name == 'mlqa':
            data_dir = join(self.config['download_dir'], f'mlqa/MLQA_V1/test')
            data_file = f'test-context-{lang}-question-{lang}.json'
        else:
            data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-dev.txt')
            data_file = f'tydiqa.{lang}.dev.txt.json'

        examples, features, dataset = self._get_data(dataset_name, partition, lang, data_dir, data_file,
                                                     cl_model=cl_model, model_path=model_path)
        return dataset if only_dataset else (examples, features, dataset)

    def get_target(self, dataset_name, partition, lang, only_dataset=False, cl_model=None, model_path=None):
        assert dataset_name in ['xquad', 'mlqa', 'tydiqa']
        if dataset_name == 'xquad':
            data_dir = join(self.config['download_dir'], 'xquad')
            data_file = f'xquad.{lang}.json'
        elif dataset_name == 'mlqa':
            data_dir = join(self.config['download_dir'], f'mlqa/MLQA_V1/{partition}')
            data_file = f'{partition}-context-{lang}-question-{lang}.json'
        else:
            data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-dev.txt')
            data_file = f'tydiqa.{lang}.dev.txt.json'
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, lang)
        if '_mlqa' in cache_dataset_path:
            cache_dataset_path = cache_dataset_path.replace('_mlqa', '')

        if only_dataset and os.path.exists(cache_dataset_path):
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info('Loaded dataset from cache')
            return dataset

        data_cand = None
        if cl_model is not None:
            data_dir_cand = join(self.config['download_dir'], 'meta_data')
            data_cand_file = 'candidate_data.bin'
            data_cand = join(data_dir_cand, data_cand_file)
        else:
            data_dir_cand = join(self.config['download_dir'], 'meta_data')
            data_cand_file = 'candidate_data.bin'
            data_cand = join(data_dir_cand, data_cand_file)
        examples, features, dataset = self._get_data(dataset_name, partition, lang, data_dir, data_file,
                                                     cl_model=cl_model, model_path=model_path)
        return dataset if only_dataset else (examples, features, dataset)

    def get_cache_feature_path(self, dataset_name, partition, lang):
        model_type = self.config['model_type']
        cache_dir = join(self.config['data_dir'], f'cache/{dataset_name}/{model_type}')
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f'{partition}.{lang}.{self.max_seg_len}.1'  # {self.config.meta_example_num}
        cache_path = join(cache_dir, f'{cache_name}.bin')
        return cache_path

    def get_cache_feature_path_norm(self, dataset_name, partition, lang):
        model_type = self.config['model_type']
        cache_dir = join(self.config['data_dir'], f'cache/{dataset_name}/{model_type}')
        os.makedirs(cache_dir, exist_ok=True)
        path = self.config.path_ckpt.split("/")[-2]
        cache_name = f'{partition}.{lang}.{self.max_seg_len}.{path}'  # {self.config.meta_example_num}
        cache_path = join(cache_dir, f'{cache_name}.bin')
        return cache_path

    def get_cache_dataset_path(self, dataset_name, partition, lang):
        cache_path = self.get_cache_feature_path(dataset_name, partition, lang)
        cache_path = cache_path[:-4] + '.dataset'
        return cache_path
