# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = "1,0,2,3"
import numpy as np
import numpy
import torch
# from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import sys

sys.path.append("../src/NER")
from seqeval.metrics.sequence_labeling import precision_score, recall_score, f1_score
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file

import learn2learn as l2l
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    XLMConfig,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification
)
from xlm import XLMForTokenClassification

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "mbert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    model.to('cuda:0')
    args.train_batch_size = args.batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    best_score = 0.0
    best_checkpoint = None
    patience = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Add here for reproductibility (even between python 2 and 3)

    meta_train_error = 0.0
    step = 0
    max_f1 = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            qry_inputs = [{
                "input_ids": batch[4][i],
                "attention_mask": batch[5][i],
                "token_type_ids": batch[6][i] if args.model_type in ["bert", "xlnet"] else None,
                "labels": batch[7][i],
            } for i in range(0, args.batch_size)]

            spt_inputs = [{
                "input_ids": batch[0][i],
                "attention_mask": batch[1][i],
                "token_type_ids": batch[2][i] if args.model_type in ["bert", "xlnet"] else None,
                "labels": batch[3][i],
            } for i in range(0, args.batch_size)]

            maml = l2l.algorithms.MAML(model, lr=scheduler.get_lr()[0], first_order=True)
            # loss_qry_all = 0.0
            for j in range(args.batch_size):  # bachsize
                learner = maml.clone()
                for _ in range(0, args.meta_steps):  # meta epoch
                    outputs = learner(**spt_inputs[j])
                    loss = outputs[0]

                    loss = loss.mean()
                    learner.adapt(loss, allow_nograd=True, allow_unused=True)

                # On the query data
                loss_qry = learner(**qry_inputs[j])[0].mean()
                # loss_qry_all += loss_qry
                loss_qry.backward()

                meta_train_error += loss_qry.item()

            # Average the accumulated gradients and optimize
            # loss_qry_all = loss_qry_all / args["batch_size"]
            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / args.batch_size)
            # loss_qry_all.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:
                    # Only evaluate on single GPU otherwise metrics may not average well
                    results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                          lang=args.train_langs, lang2id=lang2id)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.save_only_best_checkpoint:
                    result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                         prefix=global_step, lang=args.train_langs, lang2id=lang2id)
                    if result["f1"] > best_score:
                        logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
                        best_score = result["f1"]
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-best" + (
                            "" if args.train_langs == 'en' else f"_{args.train_langs}"))
                        best_checkpoint = output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving the best model checkpoint to %s", output_dir)
                        logger.info("Reset patience to 0")
                        patience = 0
                    else:
                        patience += 1
                        logger.info("Hit patience={}".format(patience))
                        if args.eval_patience > 0 and patience > args.eval_patience:
                            logger.info("early stop! patience={}".format(patience))
                            epoch_iterator.close()
                            train_iterator.close()
                            if args.local_rank in [-1, 0]:
                                tb_writer.close()
                            return global_step, tr_loss / global_step
                else:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step) + (
                        "" if args.train_langs == 'en' else f"_{args.train_langs}"))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            # Save the best model checkpoint
        output_dir = os.path.join(args.output_dir,
                                  "checkpoint-final" + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))
        # best_checkpoint = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving the final model checkpoint to %s", output_dir)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", lang2id=None,
             print_result=True):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang,
                                           lang2id=lang2id)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #  model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
            if args.model_type == 'xlm':
                inputs["langs"] = batch[4]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    if nb_eval_steps == 0:
        results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
    else:
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }

    if print_result:
        logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def build_support_features(base_features, target_features=None, support_size=2, cl_model=None, seed=None):
    if target_features == None:
        target_features = base_features
    if seed:
        random.seed(seed)
        spt_len = [i for i in range(len(base_features))]

        for i in range(len(target_features)):
            target_features[i].rankings_spt = np.array(random.sample(spt_len, 10))
        return target_features
    TOP_K = support_size
    # pdist = nn.PairwiseDistance(p=2)
    target_reprs = np.stack([numpy.array(item.representation) for item in target_features])
    base_reprs = np.stack([numpy.array(item.representation) for item in base_features])  # sample_num x feature_dim

    # compute pairwise cosine distance
    dis = np.matmul(target_reprs, base_reprs.T)  # target_num x base_num

    base_norm = np.linalg.norm(base_reprs, axis=1)  # base_num
    base_norm = np.stack([base_norm] * len(target_features), axis=0)  # target_num x base_num

    dis = dis / base_norm  # target_num x base_num
    relevance = np.argsort(dis, axis=1)
    if cl_model is not None:
        support_size = 64
    for i, item in tqdm(enumerate(target_features), desc='[get meta data]'):
        chosen_ids = relevance[i][-1 * (support_size + 1): -1]
        # logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
        # support = [base_features[id] for id in chosen_ids]
        # support_set.append(support)
        if cl_model is not None:
            chosen_dict = {}
            target_features[i].meta_set_num = []
            target_representation = torch.tensor([target_features[i].input_rep], dtype=torch.float).repeat(support_size,
                                                                                                           1, 1).to(
                cl_model.device)
            target_attention = torch.tensor([target_features[i].attention], dtype=torch.float).repeat(support_size,
                                                                                                      1).to(
                cl_model.device)
            target_latent_representation = torch.matmul(target_representation, cl_model.probe.proj)

            chosen_representation = torch.tensor([base_features[id].input_rep for id in chosen_ids],
                                                 dtype=torch.float).to(cl_model.device)
            chosen_attention = torch.tensor([base_features[id].attention for id in chosen_ids], dtype=torch.float).to(
                cl_model.device)
            chosen_latent_representation = torch.matmul(chosen_representation, cl_model.probe.proj)
            score = cl_model.get_wmd(target_attention, chosen_attention, target_latent_representation,
                                     chosen_latent_representation).tolist()
            for i_, id in enumerate(chosen_ids):
                if id not in chosen_dict:
                    chosen_dict[id] = score[i_]
            chosen_dict = sorted(chosen_dict.items(), key=lambda x: x[1])
            for pos in chosen_dict[:TOP_K]:
                pos = int(pos[0])
                try:
                    target_features[i].rankings_spt.append(pos)
                except:
                    target_features[i].rankings_spt = [pos]
            target_features[i].rankings_spt = np.array(target_features[i].rankings_spt)
        else:
            target_features[i].rankings_spt = np.array(chosen_ids)
    return target_features


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
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
            data_file = os.path.join(args.data_dir, lg, "{}".format(mode if lg == 'en' else 'dev'))
            logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
            examples = read_examples_from_file(data_file, lg, lang2id)
            features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                       is_training=(mode == 'train'),
                                                       cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                       cls_token=tokenizer.cls_token,
                                                       cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                       sep_token=tokenizer.sep_token,
                                                       sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                       pad_on_left=bool(args.model_type in ["xlnet"]),
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                           0],
                                                       pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
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
    if args.local_rank == 0 and not evaluate:
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


def load_and_cache_examples_meta(args, tokenizer, labels, pad_token_label_id, mode, lang,
                                 lang2id=None, few_shot=-1, sbert=True, ner_model=None, cl_model=None):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, 'en_task',
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        spt_features = torch.load(cached_features_file)
    else:
        logger.info("all languages = {}".format(lang))
        spt_features = []
        data_file = os.path.join(args.data_dir, 'en', "train")
        logger.info("Creating features from dataset file at {} in language {}".format(data_file, 'en'))
        examples = read_examples_from_file(data_file, 'en', lang2id)
        spt_features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                    is_training=(mode == 'train'),
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                        0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id,
                                                    lang='en',
                                                    sbert=sbert,
                                                   ner_model=ner_model)
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(spt_features)))
            torch.save(spt_features, cached_features_file)

    langs = lang.split(',')
    logger.info("all languages = {}".format(lang))
    features = []
    for lg in langs:
        data_file = os.path.join(args.data_dir, lg, "{}".format(mode))
        logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
        examples = read_examples_from_file(data_file, lg, lang2id)
        features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                   is_training=(mode == 'train'),
                                                   cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                   cls_token=tokenizer.cls_token,
                                                   cls_token_segment_id=2 if args.model_type in [
                                                       "xlnet"] else 0,
                                                   sep_token=tokenizer.sep_token,
                                                   sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                   pad_on_left=bool(args.model_type in ["xlnet"]),
                                                   pad_token=
                                                   tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                       0],
                                                   pad_token_segment_id=4 if args.model_type in [
                                                       "xlnet"] else 0,
                                                   pad_token_label_id=pad_token_label_id,
                                                   lang=lg,
                                                   spt_feature=spt_features,
                                                   sbert=sbert,
                                                   ner_model=ner_model,
                                                   cl_model=cl_model
                                                   )
        features.extend(features_lg)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    qry_features = build_support_features(spt_features, target_features=features, support_size=2, cl_model=None, seed=args.sample_seed)
    random.seed(args.seed)
    l_input_ids_s = []
    l_attention_masks_s = []
    l_token_type_ids_s = []
    l_label_ids_s = []

    l_input_ids_q = []
    l_attention_masks_q = []
    l_token_type_ids_q = []
    l_label_ids_q = []
    if mode == 'train' and args.meta_num < 500:
        qry_features = qry_features[:args.meta_num]
    random.shuffle(qry_features)
    dic_num = {}
    for index in range(len(qry_features)):
        spt = qry_features[index].rankings_spt[0]
        if spt not in dic_num.keys():
            dic_num[spt] = 1
        else:
            dic_num[spt] += 1
    dic_num = sorted(dic_num.items(), key=lambda d: d[1], reverse=True)
    print(dic_num)
    print(len(dic_num))

    s = 0
    print("META Batching")
    for _ in tqdm(range(args.batch_sz)):
        l_input_ids_spt = []
        l_attention_masks_spt = []
        l_token_type_ids_spt = []
        l_all_label_ids_spt = []

        l_input_ids_qry = []
        l_attention_masks_qry = []
        l_token_type_ids_qry = []
        l_all_label_ids_qry = []

        # Pick q_qry from qry examples randomly
        if s + args.q_qry < len(qry_features):
            qry_indices = range(s, s + args.q_qry)
            s = s + args.q_qry
        elif s < len(qry_features):
            t = s + args.q_qry - len(qry_features)
            qry_indices = list(range(s, len(qry_features))) + list(range(0, t))
            s = t
        else:
            s = 0
            qry_indices = range(s, len(qry_features))

        all_feature_index = torch.tensor(qry_indices, dtype=torch.long)
        # l_feature_index.append(all_feature_index)

        # random.sample(len(qry_examples), k=data_args.q_qry)

        for index in qry_indices:
            l_input_ids_qry.append(qry_features[index].input_ids)
            l_attention_masks_qry.append(qry_features[index].input_mask)
            l_token_type_ids_qry.append(qry_features[index].segment_ids)
            l_all_label_ids_qry.append(qry_features[index].label_ids)

            spt_indices = qry_features[index].rankings_spt[:args.meta_example_num]
            for spt_index in spt_indices:
                l_input_ids_spt.append(spt_features[spt_index].input_ids)
                l_attention_masks_spt.append(spt_features[spt_index].input_mask)
                l_token_type_ids_spt.append(spt_features[spt_index].segment_ids)
                l_all_label_ids_spt.append(spt_features[spt_index].label_ids)

        all_input_ids = torch.tensor(l_input_ids_spt, dtype=torch.long)
        l_input_ids_s.append(all_input_ids)
        l_attention_masks_s.append(torch.tensor(l_attention_masks_spt, dtype=torch.long))
        l_token_type_ids_s.append(torch.tensor(l_token_type_ids_spt, dtype=torch.long))
        l_label_ids_s.append(torch.tensor(l_all_label_ids_spt, dtype=torch.long))

        all_input_ids = torch.tensor(l_input_ids_qry, dtype=torch.long)
        l_input_ids_q.append(all_input_ids)
        l_attention_masks_q.append(torch.tensor(l_attention_masks_qry, dtype=torch.long))
        l_token_type_ids_q.append(torch.tensor(l_token_type_ids_qry, dtype=torch.long))
        l_label_ids_q.append(torch.tensor(l_all_label_ids_qry, dtype=torch.long))

    # Convert to Tensors and build dataset
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s), torch.stack(l_label_ids_s),
                                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q), torch.stack(l_token_type_ids_q), torch.stack(l_label_ids_q),
                                all_langs)
    else:
        dataset = TensorDataset(torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s), torch.stack(l_label_ids_s),
                                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q), torch.stack(l_token_type_ids_q), torch.stack(l_label_ids_q))
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='../../data/wikiann/', type=str,
                        help="The input data dir. Should contain the training files for the NER/POS task.")
    parser.add_argument("--model_type", default='mbert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + "bert-base-multilingual-cased, xlm-roberta-large")
    parser.add_argument("--output_dir", default='../../output/NER/output_mbert_sample/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="../../data/wikiann/label.txt", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_predict_dev", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--init_checkpoint",
                        default='../../output/NER/output_mbert/checkpoint-final/', type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--few_shot", default=-1, type=int,
                        help="num of few-shot exampes")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=30, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--q_qry", default=8, type=int)
    parser.add_argument("--meta_example_num", default=1, type=int)
    parser.add_argument("--meta_steps", default=3, type=int)
    parser.add_argument("--batch_sz", default=500, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best_checkpoint", action="store_true",
                        help="Save only the best checkpoint during training")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=111,
                        help="random seed for initialization")

    parser.add_argument("--sample_seed", type=int, default=40,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--predict_langs", type=str, default="en, ar, bn, fi, id, ko, ru, sw, te, zh",
                        help="prediction languages")
    parser.add_argument("--train_langs", default="ar", type=str,
                        help="The languages in the training sets.")
    parser.add_argument("--log_file", type=str, default='../../output/NER/log.txt', help="log file")
    parser.add_argument("--eval_patience", type=int, default=-1,
                        help="wait N times of decreasing dev score before early stop during training")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare NER/POS task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.init_checkpoint:
        logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
        model = model_class.from_pretrained(args.init_checkpoint,
                                            config=config,
                                            cache_dir=args.init_checkpoint)
    else:
        logger.info("loading from cached model = {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("Using lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples_meta(args, tokenizer, labels, pad_token_label_id, mode="train",
                                                     lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot,
                                                     sbert=False, ner_model=model, cl_model=None)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use default names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Initialization for evaluation
    results = {}
    if os.path.exists(os.path.join(args.output_dir,
                                   'checkpoint-final' + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-final' + (
            "" if args.train_langs == 'en' else f"_{args.train_langs}"))
    elif os.path.exists(os.path.join(args.output_dir,
                                     'checkpoint-best' + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))):
        best_checkpoint = os.path.join(args.output_dir,
                                       'checkpoint-best' + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step,
                                 lang=args.train_langs, lang2id=lang2id)
            if result["f1"] > best_f1:
                best_checkpoint = checkpoint
                best_f1 = result["f1"]
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

    # Prediction
    all_f1 = ""

    f = open('result_sample.txt', 'a+', encoding='utf-8')
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "a") as result_writer:
            for lang in args.predict_langs.split(', '):
                if not os.path.exists(os.path.join(args.data_dir, lang, 'test')):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test",
                                               lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                all_f1 += "%.2f " % (result['f1'] * 100)
                # Save predictions
                # output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
                # infile = os.path.join(args.data_dir, lang, "test")
                # idxfile = infile + '.idx'
                # save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)
            print(all_f1, file=f)
    # Predict dev set
    if args.do_predict_dev and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "dev_results.txt")
        with open(output_test_results_file, "w") as result_writer:
            for lang in args.predict_langs.split(','):
                if not os.path.exists(os.path.join(args.data_dir, lang, 'dev.{}'.format(args.model_name_or_path))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                               lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                # Save predictions
                output_test_predictions_file = os.path.join(args.output_dir, "dev_{}_predictions.txt".format(lang))
                infile = os.path.join(args.data_dir, lang, "dev.{}".format(args.model_name_or_path))
                idxfile = infile + '.idx'
                save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)


def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
    # Save predictions
    with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
        text = text_reader.readlines()
        index = idx_reader.readlines()
        assert len(text) == len(index)

    # Sanity check on the predictions
    with open(output_file, "w") as writer:
        example_id = 0
        prev_id = int(index[0])
        for line, idx in zip(text, index):
            if line == "" or line == "\n":
                example_id += 1
            else:
                cur_id = int(idx)
                output_line = '\n' if cur_id != prev_id else ''
                if output_word_prediction:
                    output_line += line.split()[0] + '\t'
                output_line += predictions[example_id].pop(0) + '\n'
                writer.write(output_line)
                prev_id = cur_id


if __name__ == "__main__":
    main()

    # --do_train --do_eval
