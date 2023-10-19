import argparse
import logging
import pickle
import random
import sys
from copy import deepcopy

from src.SDMM import train_helper
from src.SDMM.models import prob_wmd

import time
from os.path import join
import os
import learn2learn as l2l

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, BertForQuestionAnswering, \
    XLMRobertaForQuestionAnswering, XLMForQuestionAnswering

import util
from evaluate_mlqa import evaluate as mlqa_evaluate
from evaluate_squad import evaluate as squad_evaluate
from squad import SquadResult
from squad_metrics import compute_predictions_logits
from tensorize_mrc import QaDataProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


# model_types_wo_token_type_ids = ["xlm", "roberta", "distilbert", "camembert", "bart", "mt5"]

def get_QA_model(config):
    if 'bert' in config['model_type']:
        return BertForQuestionAnswering.from_pretrained(config['pretrained'])
    elif 'xlmr' in config['model_type'] or "infoxlm" in config['model_type']:
        return XLMRobertaForQuestionAnswering.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlm':
        return XLMForQuestionAnswering.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])


class QaRunner:
    def __init__(self, config_name, gpu_id=0, seed=None, saved_suffix=None, model_path=None, sub_name='',
                 config_file_path="mrc_experiments.conf", cl_model=False, squad_mrc_similarity=False):
        self.name = config_name
        if squad_mrc_similarity:
            self.model_path_squad = model_path
        else:
            self.model_path_squad = None
        self.optimizer_model_path = model_path
        if seed is None:
            seed = random.randint(1, 1000)
        self.seed = seed
        # self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.name_suffix = "{}_seed_{}".format(self.name, self.seed)
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name, sub_name, config_file_path)
        self.config.gpu_id = gpu_id
        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info(f'Log file path: {log_path}')

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = QaDataProcessor(self.config)
        self.model = self.initialize_model(saved_suffix, model_path)
        if cl_model:
            m = torch.load(self.config.path_ckpt, map_location=torch.device('cpu'))
            args = m['config']
            args.gpu_id = self.config.gpu_id
            e = train_helper.experiment(args, args.save_file_path)
            self.cl_model = prob_wmd(experiment=e)
            self.cl_model.load_state_dict(m["state_dict"], strict=False)
        else:
            self.cl_model = None

        if self.config.freeze_layer >= 0:
            if self.config.freeze_layer == 0:
                no_grad_param_names = ['embeddings']  # layer.0
            else:
                no_grad_param_names = ['embeddings', 'pooler'] + ['layer.{}.'.format(i) for i in
                                                                  range(self.config.freeze_layer + 1)]
            logger.info("The frozen parameters are:")
            for name, param in self.model.named_parameters():
                if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
                    param.requires_grad = False
                    logger.info("  {}".format(name))

    def initialize_model(self, saved_suffix=None, model_path=None):
        # model = TransformerQa(self.config)
        model = get_QA_model(self.config)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        if model_path:
            file_name = model_path.split('_seed')[0]
            model_path = join(f'../../data/{file_name}/',
                             f'model_{model_path}/pytorch_model.bin')
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model

    def prepare_inputs_meta(self, batch, is_training):
        input_list = []
        for i in range(batch[0].size(0)):
            inputs = {
                'input_ids': batch[0][i],
                'attention_mask': batch[1][i],
                'token_type_ids': batch[2][i],
                'meta_input_ids': batch[7][i],
                'meta_attention_mask': batch[8][i],
                'meta_token_type_ids': batch[9][i]
            }
            if is_training:
                inputs.update({
                    'start_positions': batch[3][i],
                    'end_positions': batch[4][i],
                    'meta_start_positions': batch[10][i],
                    'meta_end_positions': batch[11][i]
                })
            input_list.append(inputs)
        return input_list

    def prepare_inputs(self, batch, is_training):
        inputs = {
            'input_ids': batch[0].squeeze(1),
            'attention_mask': batch[1].squeeze(1),
            'token_type_ids': batch[2].squeeze(1),
        }
        if is_training:
            inputs.update({
                'start_positions': batch[3].squeeze(1),
                'end_positions': batch[4].squeeze(1),
            })
        return inputs

    def inner_update(self, data_support, inner_opt, inner_steps, scheduler_meta=None):
        self.model.train()
        loss_all = 0.0
        for i in range(inner_steps):
            inner_opt.zero_grad()
            output = self.model(**data_support)
            loss = output[0]
            loss.backward()
            inner_opt.step()
            self.model.zero_grad()
            if scheduler_meta:
                scheduler_meta.step()
            loss_all += loss.item()

        return loss_all

    def train(self, lang='en'):
        conf = self.config
        logger.info(conf)
        epochs, batch_size, grad_accum = conf['num_epochs'], conf['batch_size'], conf['gradient_accumulation_steps']

        self.model.to(self.device)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        train_dataset = self.data.get_meta(conf['train_dataset'], 'train', lang=lang, only_dataset=True,
                                           cl_model=self.cl_model)
        dev_examples, dev_features, dev_dataset = self.data.get_source(conf['train_dataset'], 'dev.txt', lang=lang,
                                                                       model_path=self.model_path_squad)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                      batch_size=batch_size, drop_last=False)

        # Set up optimizer and scheduler
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        optimizer = self.get_optimizer(self.model)
        file_name = self.optimizer_model_path.split('_seed')[0]
        path_ckpt = join(f'../../data/{file_name}/',
                         f'model_{self.optimizer_model_path}/optimizer.pt')
        optimizer.load_state_dict(torch.load(path_ckpt, map_location=torch.device(self.device)))
        scheduler = self.get_scheduler(optimizer, total_update_steps)

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(train_dataset))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_history = []  # Full history of effective loss; length equals total update steps
        max_em = 0
        start_time = time.time()
        # self.model.zero_grad()
        meta_train_error = 0.0
        step = 0
        max_f1 = 0
        for epo in range(epochs):
            print("epoch: ", epo)
            # per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector(epo)
            for batch in train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                qry_inputs = [{
                    "input_ids": batch[0][i],
                    "attention_mask": batch[1][i],
                    "token_type_ids": None if conf.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2][i],
                    "start_positions": batch[3][i],
                    "end_positions": batch[4][i],
                } for i in range(0, conf["batch_size"])]

                spt_inputs = [{
                    "input_ids": batch[7][i],
                    "attention_mask": batch[8][i],
                    "token_type_ids": None if conf.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[9][i],
                    "start_positions": batch[10][i],
                    "end_positions": batch[11][i],
                } for i in range(0, conf["batch_size"])]

                self.model.train()

                maml = l2l.algorithms.MAML(self.model, lr=scheduler.get_lr()[0], first_order=True)
                # loss_qry_all = 0.0
                for j in range(conf["batch_size"]):  # bachsize
                    learner = maml.clone()

                    for _ in range(0, conf["meta_steps"]):  # meta epoch
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
                # loss_qry_all = loss_qry_all / conf["batch_size"]
                for p in maml.parameters():
                    if p.grad is not None:
                        p.grad.mul_(1.0 / conf["batch_size"])
                # loss_qry_all.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                # Report
                if step % conf['report_frequency'] == 0:
                    # Show avg loss during last report interval
                    avg_loss = meta_train_error
                    meta_train_error = 0.0
                    end_time = time.time()
                    logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                (step, avg_loss, conf['report_frequency'] / (end_time - start_time)))
                    start_time = end_time

                # Evaluate
                if step > 0 and step % conf['eval_frequency'] == 0:
                    metrics, _ = self.evaluate(dev_examples, dev_features, dev_dataset, step,
                                               tb_writer)
                    print(metrics)
                    if metrics['exact_match'] >= max_em:
                        if metrics['exact_match'] == max_em and max_f1 > metrics['f1']:
                            pass
                        else:
                            max_em = metrics['exact_match']
                            max_f1 = metrics['f1']
                            # max_em = metrics['exact_match']
                            self.save_model_checkpoint(self.model, self.name_suffix + '_' + lang)
                    logger.info(f'Eval max em: {max_em:.2f}')
                    start_time = time.time()
                step += 1

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % step)

        # Eval at the end
        metrics, _ = self.evaluate(dev_examples, dev_features, dev_dataset, step, tb_writer)
        print(metrics)
        if metrics['exact_match'] >= max_em:
            if metrics['exact_match'] == max_em and max_f1 > metrics['f1']:
                pass
            else:
                max_em = metrics['exact_match']
                # max_f1 = metrics['f1']
                self.save_model_checkpoint(self.model, self.name_suffix + '_' + lang)
        logger.info(f'Eval max em: {max_em:.2f}')

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, examples, features, dataset, step=0, tb_writer=None, output_results_file=None,
                 output_prediction_file=None, output_nbest_file=None, output_null_log_odds_file=None,
                 dataset_name=None, lang=None, verbose_logging=False):
        conf = self.config
        logger.info(f'Step {step}: evaluating on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=conf['batch_size'],
                                pin_memory=False, num_workers=0, persistent_workers=False)

        self.model.eval()
        self.model.to(self.device)
        results = []
        for batch in dataloader:
            feature_indices = batch[3]  # To identify feature in batch eval
            inputs = self.prepare_inputs(batch, is_training=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            inputs.update({'return_dict': False})
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Build results from batch output
            for i, feature_idx in enumerate(feature_indices):
                feature = features[feature_idx.item()]
                feature_unique_id = int(feature.unique_id)
                feature_output = [output[i].tolist() for output in outputs]

                start_logits, end_logits = feature_output[:2]
                result = SquadResult(feature_unique_id, start_logits, end_logits)
                results.append(result)

        if output_results_file:
            with open(output_results_file, 'wb') as f:
                pickle.dump(results, f)

        # Evaluate
        metrics, predictions = self.evaluate_from_results(examples, features, results, output_prediction_file,
                                                          output_nbest_file, output_null_log_odds_file,
                                                          dataset_name, lang, verbose_logging)
        if tb_writer:
            for name, val in metrics.items():
                tb_writer.add_scalar(f'Train_Eval_{name}', val, step)
        return metrics, predictions

    def evaluate_meta(self, examples, features, dataset, step=0, tb_writer=None, output_results_file=None,
                      output_prediction_file=None, output_nbest_file=None, output_null_log_odds_file=None,
                      dataset_name=None, lang=None, verbose_logging=False):

        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)
        conf = self.config
        logger.info(f'Step {step}: evaluating on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=conf['batch_size_meta'],
                                pin_memory=False, num_workers=0, persistent_workers=False)

        self.model.eval()
        self.model.to(self.device)
        results = []
        self.config.meta_learning_rate = 1e-5
        optimizer_meta = self.get_meta_optimizer(self.model)
        for batch in dataloader:
            feature_indices = batch[3]  # To identify feature in batch eval
            input_list = self.prepare_inputs_meta(batch, is_training=True)
            for inputs in input_list:
                inputs_meta = {k.replace("meta_", ""): v.to(self.device) for k, v in inputs.items() if "meta" in k}
                self.inner_update(inputs_meta, optimizer_meta, self.config.meta_steps)

            inputs = self.prepare_inputs(batch, is_training=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs.update({'return_dict': False})
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Build results from batch output
            for i, feature_idx in enumerate(feature_indices):
                feature = features[feature_idx.item()]
                feature_unique_id = int(feature.unique_id)
                feature_output = [output[i].tolist() for output in outputs]

                start_logits, end_logits = feature_output[:2]
                result = SquadResult(feature_unique_id, start_logits, end_logits)
                results.append(result)
            self.load_weights(names, weights)

        if output_results_file:
            with open(output_results_file, 'wb') as f:
                pickle.dump(results, f)

        # Evaluate
        metrics, predictions = self.evaluate_from_results(examples, features, results, output_prediction_file,
                                                          output_nbest_file, output_null_log_odds_file,
                                                          dataset_name, lang, verbose_logging)
        if tb_writer:
            for name, val in metrics.items():
                tb_writer.add_scalar(f'Train_Eval_{name}', val, step)
        return metrics, predictions

    def evaluate_from_results(self, examples, features, results, output_prediction_file=None, output_nbest_file=None,
                              output_null_log_odds_file=None, dataset_name=None, lang=None, verbose_logging=False):
        conf = self.config
        predictions = compute_predictions_logits(examples, features, results, conf['n_best_predictions'],
                                                 conf['max_answer_len'], False, output_prediction_file,
                                                 output_nbest_file, output_null_log_odds_file,
                                                 conf['version_2_with_negative'], conf['null_score_diff_threshold'],
                                                 self.data.get_tokenizer(), verbose_logging)
        if dataset_name == 'mlqa':
            metrics = mlqa_evaluate(examples, predictions, lang)
        else:
            metrics, wrong_examples = squad_evaluate(examples, predictions)
            """json_str = json.dumps(wrong_examples, indent=4, ensure_ascii=False)
            with open('wrong_results_{}_{}.json'.format(dataset_name, lang), 'w', encoding="utf-8") as json_file:
                json_file.write(json_str)"""
        return metrics, predictions

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_param = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(grouped_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps'])
        return optimizer

    def get_meta_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_param = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(grouped_param, lr=self.config.meta_learning_rate, eps=self.config['adam_eps'])
        return optimizer

    def get_scheduler(self, optimizer, total_update_steps):
        if self.config['model_type'] == 'mt5':
            # scheduler = get_constant_schedule(optimizer)
            cooldown_start = int(total_update_steps * 0.7)

            def lr_lambda(current_step: int):
                return 1 if current_step < cooldown_start else 0.3

            return LambdaLR(optimizer, lr_lambda, -1)
        else:
            warmup_steps = int(total_update_steps * self.config['warmup_ratio'])
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_update_steps)
        return scheduler

    def save_model_checkpoint(self, model, name_suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{name_suffix}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)

    def get_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params

    def load_weights(self, names, params):
        model_params = self.model.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)


def get_all_score_sum(all_list, one_seed_list):
    if not all_list:
        return [[i] for i in one_seed_list]
    for i, a in enumerate(one_seed_list):
        all_list[i].append(a)
    return all_list


import cross_lingual_mrc_test


def main(ml_type, train_num, lang='en', sub_name='', config_file_path="mrc_experiments.conf", xquad_test=True,
         mlqa_test=True, tydiqa_test=True, squad_mrc_similarity=False, cl_model=False):
    args = argparse.ArgumentParser(
        description='Paraphrase using PyTorch')
    args.use_cuda = torch.cuda.is_available()
    args.ml_type = ml_type
    all_f1_xquad = []
    all_em_xquad = []
    all_f1_mlqa = []
    all_em_mlqa = []
    all_f1_tydiqa = []
    all_em_tydiqa = []
    seed_list = [111, 222, 3333, 4444, 444]
    # seed_list = [人、、、、、、、、、、、、、、、、«
    train = False
    for seed in seed_list:
        if args.use_cuda:
            gpu_id = 3
        else:
            gpu_id = None
        # mrc
        config_name = "{}_zero_shot".format(args.ml_type)
        runner = QaRunner(config_name, gpu_id, seed=seed, sub_name=sub_name, config_file_path=config_file_path,
                          model_path=f'{config_name}_seed_{seed}', squad_mrc_similarity=squad_mrc_similarity,
                          cl_model=cl_model)
        # model = runner.initialize_model()
        # seed_list.append(runner.seed)

        model_path = join(runner.config['log_dir'], f'model_{runner.name_suffix}_{lang}.bin')
        if not os.path.exists(model_path) or train:
            runner.train(lang)

        # cross-lingual mrc test
        config_name, saved_suffix = "{}_zero_shot".format(args.ml_type), runner.name_suffix
        saved_suffix += "_" + lang
        if xquad_test:
            dataset_name = "xquad"
            evaluator = cross_lingual_mrc_test.Evaluator_meta(config_name, saved_suffix, gpu_id, seed=runner.seed,
                                                              sub_name=sub_name,
                                                              config_file_path=config_file_path)
            xquad_f1, xquad_em = evaluator.evaluate_task(dataset_name)
            all_f1_xquad = get_all_score_sum(all_f1_xquad, xquad_f1)
            all_em_xquad = get_all_score_sum(all_em_xquad, xquad_em)

        if mlqa_test:
            dataset_name = "mlqa"
            evaluator = cross_lingual_mrc_test.Evaluator_meta(config_name, saved_suffix, gpu_id, seed=runner.seed,
                                                              sub_name=sub_name, config_file_path=config_file_path)
            mlqa_f1, mlqa_em = evaluator.evaluate_task(dataset_name)
            all_f1_mlqa = get_all_score_sum(all_f1_mlqa, mlqa_f1)
            all_em_mlqa = get_all_score_sum(all_em_mlqa, mlqa_em)

        if tydiqa_test:
            dataset_name = "tydiqa"
            evaluator = cross_lingual_mrc_test.Evaluator_meta(config_name, saved_suffix, gpu_id, seed=runner.seed,
                                                              sub_name=sub_name, config_file_path=config_file_path)
            tydiqa_f1, tydiqa_em = evaluator.evaluate_task(dataset_name)
            all_f1_tydiqa = get_all_score_sum(all_f1_tydiqa, tydiqa_f1)
            all_em_tydiqa = get_all_score_sum(all_em_tydiqa, tydiqa_em)

        del runner

    print("seed:", seed_list)
    for all_em, all_f1 in [(all_em_xquad, all_f1_xquad), (all_em_mlqa, all_f1_mlqa), (all_em_tydiqa, all_f1_tydiqa)]:
        all = ''
        if len(all_em) == 0:
            continue
        em_all = 0
        f1_all = 0
        for em_list, f1_list in zip(all_em, all_f1):
            for _ in range(2):
                min_value = min(em_list)
                min_index = em_list.index(min_value)
                em_list.pop(min_index)
                f1_list.pop(min_index)
            em_max = max(em_list)
            em_min = min(em_list)
            em_ave = sum(em_list) / len(em_list)
            em_all += em_ave
            em_delta = max(abs(em_ave - em_max), abs(em_ave - em_min))

            f1_max = max(f1_list)
            f1_min = min(f1_list)
            f1_ave = sum(f1_list) / len(f1_list)
            f1_all += f1_ave
            f1_delta = max(abs(f1_ave - f1_max), abs(f1_ave - f1_min))

            all += "&" + "{%.2f±%.2f}" % (em_ave, em_delta) + "/{%.2f±%.2f}    " % (f1_ave, f1_delta)

        all += "&" + "{%.2f}" % (em_all / len(all_em)) + "/{%.2f}" % (f1_all / len(all_f1))
        print(all)


if __name__ == '__main__':
    # runner = QaRunner("xlmr_zero_shot", 2)

    # runner.train()
    ml_type = 'mbert'
    train_num = 5
    sub_name = 'task'
    config_file_path = "mrc_experiments.conf"
    xquad_test = False
    mlqa_test = False
    tydiqa_test = True
    lang = sys.argv[1]
    cl_model=False
    main(ml_type, train_num, lang=lang, sub_name=sub_name, config_file_path=config_file_path,
         xquad_test=xquad_test, mlqa_test=mlqa_test, tydiqa_test=tydiqa_test,
         squad_mrc_similarity=squad_mrc_similarity, cl_model=cl_model)
