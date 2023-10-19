import argparse
import logging
import os
import pickle
import sys

sys.path.append("./src/MRC/")

import time
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, BertForQuestionAnswering, BertConfig, \
    XLMRobertaForQuestionAnswering, XLMForQuestionAnswering

import util
from evaluate_mlqa import evaluate as mlqa_evaluate
from evaluate_squad import evaluate as squad_evaluate
from squad import SquadResult
from squad_metrics import compute_predictions_logits
from tensorize_mrc_PRE import QaDataProcessor

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
    def __init__(self, config_name, gpu_id=0, seed=None, saved_suffix=None, config_file="PRE_experiments.conf"):
        self.name = config_name
        self.seed = seed
        # self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.name_suffix = "{}_seed_{}".format(self.name, self.seed)

        self.gpu_id = gpu_id

        # Set up config
        self.config = util.initialize_config(config_name, sub_name='', config_file_path=config_file)

        self.tokenizer = util.get_bert_tokenizer(self.config)

        self.model = self.initialize_model(saved_suffix=saved_suffix)  # saved_suffix=self.name_suffix if saved_suffix is None else saved_suffix

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
        self.config.gpu_id=gpu_id
        self.data = QaDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None, model_path=None):
        # model = TransformerQa(self.config)
        # config = BertConfig.from_pretrained(self.config['pretrained'])
        model = get_QA_model(self.config)
        if saved_suffix:
            path_ckpt = join(self.config['log_dir'], f'model_{saved_suffix}/pytorch_model.bin')
            model.load_state_dict(torch.load(path_ckpt, map_location = torch.device('cpu')))
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
        return model

    def prepare_inputs(self, batch, is_training):
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2]
        }
        if is_training:
            inputs.update({
                'start_positions': batch[3],
                'end_positions': batch[4]
            })
        return inputs

    def train(self, lang="en", patition='train'):
        model = self.model
        if patition == 'dev':
            name_suffix = self.name_suffix + "_" + lang
        else:
            name_suffix = self.name_suffix
        conf = self.config
        logger.info(conf)
        epochs, batch_size, grad_accum = conf['num_epochs'], conf['batch_size'], conf['gradient_accumulation_steps']
        model.to(self.device)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        if patition == 'train':
            train_dataset = self.data.get_source(conf['train_dataset'], 'train', only_dataset=True, lang=lang)
        else:
            train_dataset = self.data.get_dev_source(conf['train_dataset'], 'train', only_dataset=True, lang=lang)
        dev_examples, dev_features, dev_dataset = self.data.get_source(conf['train_dataset'], 'dev')
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                      batch_size=batch_size, drop_last=False)

        # Set up optimizer and scheduler
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        optimizer = self.get_optimizer(model)
        if patition == 'dev':
           optimizer = self.load_model_optimizer(optimizer, self.name_suffix)
        scheduler = self.get_scheduler(optimizer, total_update_steps)
        trained_params = model.parameters()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(train_dataset))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        metrics, _ = self.evaluate(model, dev_examples, dev_features, dev_dataset, 0, tb_writer)
        logger.info('Eval max f1: %.2f' % metrics['f1'])

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        # if patition =='train':
        model.zero_grad()
        for epo in range(epochs):
            print("epoch: ", epo)
            for batch in train_dataloader:
                model.train()
                inputs = self.prepare_inputs(batch, is_training=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                output = model(**inputs)
                loss = output[0]
                loss = loss.mean()
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf['max_grad_norm']:
                        torch.nn.utils.clip_grad_norm_(trained_params, conf['max_grad_norm'])
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', scheduler.get_last_lr()[0], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        metrics, _ = self.evaluate(model, dev_examples, dev_features, dev_dataset, len(loss_history), tb_writer)
                        if metrics['f1'] > max_f1:
                            max_f1 = metrics['f1']
                            self.save_model_checkpoint(model, optimizer, name_suffix)
                        logger.info(f'Eval f1: %.2f'% metrics['f1'])
                        logger.info(f'Eval max f1: {max_f1:.2f}')
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Eval at the end
        metrics, _ = self.evaluate(model, dev_examples, dev_features, dev_dataset, len(loss_history), tb_writer)
        if metrics['f1'] > max_f1:
            max_f1 = metrics['f1']
            self.save_model_checkpoint(model, optimizer, name_suffix)
        logger.info(f'Eval max f1: {max_f1:.2f}')

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, examples, features, dataset, step=0, tb_writer=None, output_results_file=None,
                 output_prediction_file=None, output_nbest_file=None, output_null_log_odds_file=None,
                 dataset_name=None, lang=None, verbose_logging=False):
        conf = self.config
        logger.info(f'Step {step}: evaluating on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=conf['batch_size'],
                                pin_memory=False, num_workers=0, persistent_workers=False)

        model.eval()
        model.to(self.device)
        results = []
        for batch in dataloader:
            feature_indices = batch[3]  # To identify feature in batch eval
            inputs = self.prepare_inputs(batch, is_training=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs.update({'return_dict': False})
            with torch.no_grad():
                outputs = model(**inputs)

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
            metrics = squad_evaluate(examples, predictions)[0]
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

    def save_model_checkpoint(self, model, optimizer, suffix):
        output_dir= join(self.config['log_dir'], f'model_{suffix}/')
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        path_ckpt = join(output_dir, f'pytorch_model.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)
        path_ckpt = join(output_dir, f'optimizer.pt')
        torch.save(optimizer.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}/pytorch_model.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location = torch.device('cuda')))
        logger.info('Loaded model from %s' % path_ckpt)
        return model

    def load_model_optimizer(self, optimizer, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}/optimizer.pt')
        optimizer.load_state_dict(torch.load(path_ckpt, map_location = torch.device(self.device)))
        logger.info('Loaded optimizer from %s' % path_ckpt)
        return optimizer


def get_all_score_sum(all_list, one_seed_list):
    all_list_new = []
    if not all_list:
        return one_seed_list
    for a,b in zip(all_list, one_seed_list):
        all_list_new.append(a+b)
    return all_list_new


if __name__ == '__main__':
    """config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = QaRunner(config_name, gpu_id)
    model = runner.initialize_model()

    runner.train(model)"""

    from src.MRC.cross_lingual_mrc_test_PRE import Evaluator

    args = argparse.ArgumentParser(
        description='Paraphrase using PyTorch')
    args.use_cuda = torch.cuda.is_available()
    args.ml_type = 'mbert'
    all_f1_xquad=[]
    all_em_xquad=[]
    all_f1_mlqa=[]
    all_em_mlqa=[]
    all_f1_tydiqa=[]
    all_em_tydiqa=[]
    seed_list = [111,222,3333,4444,444]
    new_train = False
    for seed in seed_list:
        # 111 222 333 444 555

        if args.use_cuda:
            gpu_id = 2
        else:
            gpu_id = None
        # mrc
        config_name = "{}_zero_shot".format(args.ml_type)
        runner = QaRunner(config_name, gpu_id, seed)
        model = runner.initialize_model()

        model_path = join(runner.config['log_dir'], f'model_{runner.name_suffix}/')
        if not os.path.exists(model_path) or new_train:
            runner.train()

        # cross-lingual mrc test
        config_name, saved_suffix = "{}_zero_shot".format(args.ml_type), runner.name_suffix
        #dataset_name = "xquad"
        #evaluator = Evaluator(config_name, saved_suffix, gpu_id)
        #xquad_f1, xquad_em = evaluator.evaluate_task(dataset_name)

        #dataset_name = "mlqa"
        #evaluator = Evaluator(config_name, saved_suffix, gpu_id)
        #mlqa_f1, mlqa_em = evaluator.evaluate_task(dataset_name)

        dataset_name = "tydiqa"
        evaluator = Evaluator(config_name, saved_suffix, gpu_id)
        tydiqa_f1, tydiqa_em = evaluator.evaluate_task(dataset_name)

        #all_f1_xquad = get_all_score_sum(all_f1_xquad, xquad_f1)
        #all_em_xquad = get_all_score_sum(all_em_xquad, xquad_em)
        #all_f1_mlqa = get_all_score_sum(all_f1_mlqa, mlqa_f1)
        #all_em_mlqa = get_all_score_sum(all_em_mlqa, mlqa_em)
        all_f1_tydiqa = get_all_score_sum(all_f1_tydiqa, tydiqa_f1)
        all_em_tydiqa = get_all_score_sum(all_em_tydiqa, tydiqa_em)

    for all_em, all_f1 in [(all_em_tydiqa, all_f1_tydiqa)]: #(all_em_xquad,all_f1_xquad), (all_em_mlqa, all_f1_mlqa),
        all = ''
        for i in range(len(all_em)):
            all += "&" + "{%.2f}" % (all_em[i]/len(seed_list)) + "/{%.2f}    " % (all_f1[i]/len(seed_list))
        all += "&" + "{%.2f}" % (sum(all_em) / (len(all_em)*len(seed_list))) + "/{%.2f}" % (sum(all_f1) / (len(all_f1)*len(seed_list)))
        print(all)