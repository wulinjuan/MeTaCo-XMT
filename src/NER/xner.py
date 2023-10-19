
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from transformers.tokenization_utils_base import TruncationStrategy

from src.NER.utils_tag import read_examples_from_file, convert_examples_to_features

logger = logging.getLogger(__name__)

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not mode == 'dev.txt':
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        langs = lang.split(',')
        logger.info("all languages = {}".format(lang))
        features = []
        for lg in langs:
            data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode, args.model_name_or_path))
            logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
            examples = read_examples_from_file(data_file, lg, lang2id)
            features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                       cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                       cls_token=tokenizer.cls_token,
                                                       cls_token_segment_id=2 if args.model_type in [
                                                           "xlnet"] else 0,
                                                       sep_token=tokenizer.sep_token,
                                                       sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                       pad_on_left=bool(args.model_type in ["xlnet"]),
                                                       pad_token=
                                                       tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                       pad_token_segment_id=4 if args.model_type in [
                                                           "xlnet"] else 0,
                                                       pad_token_label_id=pad_token_label_id,
                                                       lang=lg
                                                       )
            features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not mode == 'dev.txt':
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset