import os
from collections import defaultdict
from transformers import AdamW
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm.modeling_xlm import XLMModel
from torch.autograd import Variable
import sys

sys.path.append("/data/home10b/wlj/code/Meta_MRC/src")

from Syntax_distance_model_ablation.Dependency_paring import probe
from Syntax_distance_model_ablation.Dependency_paring.loss import L1DepthLoss, L1DistanceLoss
from Syntax_distance_model_ablation.decorators import auto_init_args, auto_init_pytorch

MODEL_CLASSES = {
    'mbert': BertModel,
    'xlm': XLMModel,
    'xlmr': XLMRobertaModel
}


def word_avg(input_vecs, mask):
    sum_vecs = (input_vecs * mask.unsqueeze(-1)).sum(1)
    avg_vecs = sum_vecs / mask.sum(1, keepdim=True)
    return avg_vecs


class MultiDisModel(nn.Module):
    def __init__(self, experiment):
        super(MultiDisModel, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.margin = self.expe.config.m
        self.use_cuda = self.expe.config.use_cuda
        self.device = 'cpu' if not self.expe.config.use_cuda else 'cuda:{}'.format(self.expe.config.gpu_id)

        # self.probe_depth = probe.OneWordPSDProbe(use_cuda=self.expe.config.use_cuda, model_dim=self.expe.config.edim)
        # self.probe_distance = probe.TwoWordPSDProbe(use_cuda=self.expe.config.use_cuda, model_dim=self.expe.config.edim)
        if self.expe.config.use_prob:
            self.probe = probe.PSDProbe(use_cuda=self.device, model_dim=self.expe.config.edim, probe_rank=32 if 'bert' in self.experiment.config.ml_type else 64)
            self.L1DepthLoss = L1DepthLoss()
            self.L1DistanceLoss = L1DistanceLoss()
        else:
            self.linear = nn.Linear(self.expe.config.edim, 32 if 'bert' in self.experiment.config.ml_type else 64)
            self.linear.to(self.device)
        pre_trained_model = MODEL_CLASSES[self.experiment.config.ml_type]
        self.transformer_model = pre_trained_model.from_pretrained(self.experiment.config.ml_token)
        self.transformer_model.to(self.device)

    def to_var(self, inputs):
        if self.use_cuda:
            inputs = inputs.to(self.device)
        return inputs

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_ is not None and
                                        inputs_.size else None for inputs_ in inputs]

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate)

        return opt

    def save(self, dev_avg, test_avg, epoch, iteration=None, name="best"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "dev_avg": dev_avg,
            "test_avg": test_avg,
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict):
        state_dict = torch.load(checkpointed_state_dict)["state_dict"]
        self.load_state_dict(state_dict)
        self.expe.log.info("model loaded!")

    @property
    def volatile(self):
        return not self.training


class prob_wmd(MultiDisModel):
    @auto_init_pytorch
    @auto_init_args
    def __init__(self, experiment, use_norm=False):
        super(prob_wmd, self).__init__(experiment)
        self.use_norm = use_norm

    def sent2param(self, input_ids, attention_mask):
        with torch.no_grad():
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                      'output_attentions': True, 'output_hidden_states': False, 'return_dict': True}
            encoded_layers = self.transformer_model(**inputs)
            input_vecs = encoded_layers['last_hidden_state']
            # input_vecs_mean = word_avg(input_vecs, attention_mask)
            attention = encoded_layers['attentions'][-1]
            attention = torch.index_select(attention, 2, torch.tensor([0]).to(self.device)).squeeze(2)
            attention = torch.mean(attention, dim=1)

        return input_vecs, attention

    def norm_attention(self, vec, mask):
        mask = mask.bool()
        attention = torch.norm(vec, dim=2)
        attention = torch.where(mask==True, attention, torch.max(attention, dim=1, keepdim=True)[0])
        attention = torch.max(attention, dim=1, keepdim=True)[0].repeat(1,attention.size()[1]) + 1e-8 - attention
        attention = torch.where(mask, attention, 0)
        attention = F.softmax(attention, dim=1)
        return attention

    def get_similiarity_map(self, proto, query, metric='cosine'):
        # proto: [batch_size, seq_len, emb_dim]
        # query: [batch_size, seq_len, emb_dim]
        global similarity_map
        feature_size = proto.shape[-2]  # seq_len

        if metric == 'cosine':
            proto = proto.unsqueeze(-3)  # [batch_size, 1, seq_len, emb_dim]
            query = query.unsqueeze(-2)  # [batch_size, seq_len, 1, emb_dim]
            query = query.repeat(1, 1, feature_size, 1)  # [batch_size, seq_len, seq_len, emb_dim]
            similarity_map = 1 - F.cosine_similarity(proto, query, dim=-1)  # [batch_size, seq_len, seq_len]
        if metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def get_wmd(self, weight1, weight2, proto, query):
        similarity_map = self.get_similiarity_map(proto, query)
        num_query = similarity_map.shape[0]
        # num_node = weight_1.shape[-1]

        for i in range(num_query):
            # _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
            cost_matrix = similarity_map[i, :, :].detach().cpu().numpy()
            cost, _, flow = cv2.EMD(weight1[i, :].view(-1, 1).detach().cpu().numpy(), weight2[i, :].view(-1, 1).detach().cpu().numpy(), cv2.DIST_USER, cost_matrix)

            similarity_map[i, :, :] = (similarity_map[i, :, :]) * torch.from_numpy(flow).to(self.device)

        # temperature = (self.args.temperature / num_node)
        logitis = similarity_map.sum(-1).sum(-1)  # [batch, seq_len]
        return logitis

    def forward(self, sent1, mask1, sent2, mask2, neg_sent1, neg_mask1, tree_tags, use_margin=-1):
        global ploss1, ploss2, ploss3, ploss4
        self.train()

        sent1, mask1, sent2, mask2, neg_sent1, neg_mask1 = \
            self.to_vars(sent1, mask1, sent2, mask2, neg_sent1, neg_mask1)

        s1_vecs, sent1_attention = self.sent2param(sent1, mask1)
        s2_vecs, sent2_attention = self.sent2param(sent2, mask2)
        n1_vecs, nsent1_attention = self.sent2param(neg_sent1, neg_mask1)
        loss = 0.0
        stl = 0.0
        if use_margin:
            if self.expe.config.use_prob:
                s1_probe = torch.matmul(s1_vecs, self.probe.proj)
                s2_probe = torch.matmul(s2_vecs, self.probe.proj)
                n1_probe = torch.matmul(n1_vecs, self.probe.proj)
            else:
                s1_probe = self.linear(s1_vecs)
                s2_probe = self.linear(s2_vecs)
                n1_probe = self.linear(n1_vecs)
            if self.expe.config.distance_type == 'wdm':
                if self.use_norm:
                    sent1_attention = self.norm_attention(s1_vecs, mask1)
                    sent2_attention = self.norm_attention(s2_vecs, mask2)
                    nsent1_attention = self.norm_attention(n1_vecs, neg_mask1)
                sent_wmd_pos = self.get_wmd(sent1_attention, sent2_attention, s1_probe, s2_probe)
                sent_wmd_neg = self.get_wmd(sent1_attention, nsent1_attention, s1_probe, n1_probe)

                loss += F.relu(self.margin - sent_wmd_neg + sent_wmd_pos).mean()
            elif self.expe.config.distance_type == 'cos':
                sent_cos_pos = F.cosine_similarity(s1_probe, s2_probe)
                sent_cos_neg = F.cosine_similarity(s1_probe, n1_probe)
                loss += F.relu(self.margin - sent_cos_pos + sent_cos_neg).mean()
            else:
                assert 'only wdm or cos'

        if self.expe.config.use_prob:
            # sentence1
            tree_tags_depth1, tree_tags_dis1, tree_tags_depth2, tree_tags_dis2, tree_tags_depth3, tree_tags_dis3 = tree_tags
            if torch.cuda.is_available():
                tree_tags_depth1, tree_tags_dis1 = tree_tags_depth1.to(self.device), tree_tags_dis1.to(self.device)
                tree_tags_depth2, tree_tags_dis2 = tree_tags_depth2.to(self.device), tree_tags_dis2.to(self.device)
                tree_tags_depth3, tree_tags_dis3 = tree_tags_depth3.to(self.device), tree_tags_dis3.to(self.device)
            sentence1_length = s1_vecs.size()[1]
            predict_dep1, predict_dis1 = self.probe(s1_vecs)
            loss11, _ = self.L1DepthLoss(predict_dep1, tree_tags_depth1, sentence1_length, self.device)
            # predict_dis1 = self.probe_distance(s1_vecs)
            loss12, _ = self.L1DistanceLoss(predict_dis1, tree_tags_dis1, sentence1_length, self.device)

            # sentence2
            sentence2_length = s2_vecs.size()[1]
            predict_dep2, predict_dis2 = self.probe(s2_vecs)
            loss21, _ = self.L1DepthLoss(predict_dep2, tree_tags_depth2, sentence2_length, self.device)
            # predict_dis2 = self.probe_distance(s2_vecs)
            loss22, _ = self.L1DistanceLoss(predict_dis2, tree_tags_dis2, sentence2_length, self.device)

            # sentence3
            sentence3_length = n1_vecs.size()[1]
            predict_dep3, predict_dis3 = self.probe(n1_vecs)
            loss31, _ = self.L1DepthLoss(predict_dep3, tree_tags_depth3, sentence3_length, self.device)
            # predict_dis2 = self.probe_distance(s2_vecs)
            loss32, _ = self.L1DistanceLoss(predict_dis3, tree_tags_dis3, sentence3_length, self.device)

            stl += loss11.mean() + loss12.mean() + loss21.mean() + loss22.mean() + loss31.mean() + loss32.mean()
            loss += stl

        return loss, stl

    def forward_eng(self, sent1, mask1, tree_tags):
        # self.train()
        self.transformer_model.eval()
        sent1, mask1 = self.to_vars(sent1, mask1)

        s1_vecs, _ = self.sent2param(sent1, mask1)

        tree_tags_depth, tree_tags_dis = tree_tags
        if torch.cuda.is_available():
            tree_tags_depth, tree_tags_dis = tree_tags_depth.to(self.device), tree_tags_dis.to(self.device)
        sentence1_length = s1_vecs.size()[1]
        predict_dep1, predict_dis1 = self.probe(s1_vecs)
        loss11, _ = self.L1DepthLoss(predict_dep1, tree_tags_depth, sentence1_length)
        # predict_dis1 = self.probe_distance(s1_vecs)
        loss12, _ = self.L1DistanceLoss(predict_dis1, tree_tags_dis, sentence1_length)

        return loss11.mean() + loss12.mean()

    def score(self, dataset):
        self.eval()
        loss_all = []
        for batch in tqdm(dataset, desc='[predicting]'):
            sent1, mask1, sent2, mask2, sent3, mask3 = self.to_vars(batch[0], batch[1], batch[4], batch[5], batch[8], batch[9])
            s1_vecs, sent1_attention = self.sent2param(sent1, mask1)
            s2_vecs, sent2_attention = self.sent2param(sent2, mask2)
            n1_vecs, nsent1_attention = self.sent2param(sent3, mask3)
            if self.expe.config.use_prob:
                s1_probe = torch.matmul(s1_vecs, self.probe.proj)
                s2_probe = torch.matmul(s2_vecs, self.probe.proj)
                n1_probe = torch.matmul(n1_vecs, self.probe.proj)
            else:
                s1_probe = self.linear(s1_vecs)
                s2_probe = self.linear(s2_vecs)
                n1_probe = self.linear(n1_vecs)
            if self.expe.config.distance_type == 'wdm':
                if self.use_norm:
                    sent1_attention = self.norm_attention(s1_vecs, mask1)
                    sent2_attention = self.norm_attention(s2_vecs, mask2)
                    nsent1_attention = self.norm_attention(n1_vecs, mask3)
                sent_wmd_pos = self.get_wmd(sent1_attention, sent2_attention, s1_probe, s2_probe)
                sent_wmd_neg = self.get_wmd(sent1_attention, nsent1_attention, s1_probe, n1_probe)

                loss = F.relu(self.margin - sent_wmd_neg + sent_wmd_pos).mean()
                loss_all.append(loss.mean().item())
            elif self.expe.config.distance_type == 'cos':
                sent_cos_pos = F.cosine_similarity(s1_probe, s2_probe)
                sent_cos_neg = F.cosine_similarity(s1_probe, n1_probe)
                loss = F.relu(self.margin - sent_cos_pos + sent_cos_neg).mean()
                loss_all.append(loss.mean().item())

            # loss_all.append(F.relu(self.margin - sent_wmd_neg + sent_wmd_pos).mean().item())

        return np.mean(loss_all)

    def pred(self, dataset):
        # spearmanrs_dep = []
        # spearmanrs_dis = []
        lengths_to_spearmanrs = defaultdict(list)

        lengths_to_spearmanrs1 = defaultdict(list)
        for batch in tqdm(dataset, desc='[predicting]'):
            sent1, mask1 = self.to_vars(batch[4], batch[5])
            s1_vecs, _ = self.sent2param(sent1, mask1)

            predict_dep1, predict_dis1 = self.probe(s1_vecs)

            for i in range(batch[0].size()[0]):
                length = torch.sum(batch[1][i]).item()
                labels_1s = (batch[2][i] != -1).float().to(self.device)
                predictions_masked = predict_dep1[i] * labels_1s
                labels_masked = batch[2][i].to(self.device) * labels_1s
                prediction = predictions_masked.detach().cpu().numpy()[1:length]
                label = labels_masked.detach().cpu().numpy()[1:length]
                sent_spearmanr = spearmanr(prediction, label)
                # spearmanrs_dep.append(sent_spearmanr.correlation)
                lengths_to_spearmanrs[length].append(sent_spearmanr.correlation)

                prediction = predict_dis1.detach().cpu().numpy()[i][1:length, 1:length]
                label = batch[3].numpy()[i][1:length, 1:length]
                sent_spearmanr = [spearmanr(pred, gold) for pred, gold in zip(prediction, label)]
                lengths_to_spearmanrs1[length].extend([x.correlation for x in sent_spearmanr if not math.isnan(x.correlation)])
                # spearmanrs_dis.extend([x.correlation for x in sent_spearmanr if not math.isnan(x.correlation)])
        # mean1 = np.mean(spearmanrs_dep)
        mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length]) for length in
                                         lengths_to_spearmanrs}
        mean1 = np.mean(
            [mean_spearman_for_each_length[x] for x in range(5, 51) if x in mean_spearman_for_each_length])

        mean_spearman_for_each_length1 = {length: np.mean(lengths_to_spearmanrs1[length])
                                         for length in lengths_to_spearmanrs1}
        # mean2 = np.mean(spearmanrs_dis)
        mean2 = np.mean([mean_spearman_for_each_length1[x] for x in range(5, 51) if x in mean_spearman_for_each_length1])
        return mean1, mean2

    def pred_s(self, dataset):
        self.eval()
        loss_all = []
        all_num = 0
        pos_num = 0
        neg_num = 0
        for batch in tqdm(dataset, desc='[predicting]'):
            sent1, mask1, sent2, mask2, sent3, mask3 = self.to_vars(batch[0], batch[1], batch[4], batch[5], batch[8], batch[9])
            s1_vecs, sent1_attention = self.sent2param(sent1, mask1)
            s2_vecs, sent2_attention = self.sent2param(sent2, mask2)
            n1_vecs, nsent1_attention = self.sent2param(sent3, mask3)
            if self.expe.config.use_prob:
                s1_probe = torch.matmul(s1_vecs, self.probe.proj)
                s2_probe = torch.matmul(s2_vecs, self.probe.proj)
                n1_probe = torch.matmul(n1_vecs, self.probe.proj)
            else:
                s1_probe = self.linear(s1_vecs)
                s2_probe = self.linear(s2_vecs)
                n1_probe = self.linear(n1_vecs)
            if self.expe.config.distance_type == 'wdm':
                if self.use_norm:
                    sent1_attention = self.norm_attention(s1_vecs, mask1)
                    sent2_attention = self.norm_attention(s2_vecs, mask2)
                    nsent1_attention = self.norm_attention(n1_vecs, mask3)
                sent_wmd_pos = self.get_wmd(sent1_attention, sent2_attention, s1_probe, s2_probe)
                sent_wmd_neg = self.get_wmd(sent1_attention, nsent1_attention, s1_probe, n1_probe)
                all_num += sent_wmd_neg.size()[0]
                pos = sent_wmd_pos - 1
                pos_num += torch.nonzero(pos<0).size()[0]
                neg = sent_wmd_neg - 1
                neg_num += torch.nonzero(neg>0).size()[0]
                loss = F.relu(self.margin - sent_wmd_neg + sent_wmd_pos)
                loss_all.append(loss.mean().item())
            elif self.expe.config.distance_type == 'cos':
                sent_cos_pos = F.cosine_similarity(s1_probe, s2_probe).mean(-1)
                sent_cos_neg = F.cosine_similarity(s1_probe, n1_probe).mean(-1)
                all_num += sent_cos_neg.size()[0]
                pos = sent_cos_pos + 1
                pos_num += torch.nonzero(pos>1).size()[0]
                neg = sent_cos_neg + 1
                neg_num += torch.nonzero(neg<1).size()[0]
                loss = F.relu(self.margin - sent_cos_pos + sent_cos_neg)
                loss_all.append(loss.mean().item())

            # loss_all.append(F.relu(self.margin - sent_wmd_neg + sent_wmd_pos).mean().item())
        pos_accuracy = pos_num/all_num
        neg_accuracy = neg_num/all_num
        all_accuracy = (pos_num + neg_num)/(2*all_num)
        return np.mean(loss_all), (pos_accuracy, neg_accuracy, all_accuracy)

    def pred_mbert(self, dataset):
        self.eval()
        loss_all = []
        all_num = 0
        pos_num = 0
        neg_num = 0
        for batch in tqdm(dataset, desc='[predicting]'):
            sent1, mask1, sent2, mask2, sent3, mask3 = self.to_vars(batch[0], batch[1], batch[4], batch[5], batch[8], batch[9])
            s1_probe, sent1_attention = self.sent2param(sent1, mask1)
            s2_probe, sent2_attention = self.sent2param(sent2, mask2)
            n1_probe, nsent1_attention = self.sent2param(sent3, mask3)

            if self.expe.config.distance_type == 'wdm':
                if self.use_norm:
                    sent1_attention = self.norm_attention(s1_probe, mask1)
                    sent2_attention = self.norm_attention(s2_probe, mask2)
                    nsent1_attention = self.norm_attention(n1_probe, mask3)
                sent_wmd_pos = self.get_wmd(sent1_attention, sent2_attention, s1_probe, s2_probe)
                sent_wmd_neg = self.get_wmd(sent1_attention, nsent1_attention, s1_probe, n1_probe)
                all_num += sent_wmd_neg.size()[0]
                pos = sent_wmd_pos - 1
                pos_num += torch.nonzero(pos<0).size()[0]
                neg = sent_wmd_neg - 1
                neg_num += torch.nonzero(neg>0).size()[0]
                loss = F.relu(self.margin - sent_wmd_neg + sent_wmd_pos)
                loss_all.append(loss.mean().item())
            elif self.expe.config.distance_type == 'cos':
                sent_cos_pos = F.cosine_similarity(s1_probe, s2_probe).mean(-1)
                sent_cos_neg = F.cosine_similarity(s1_probe, n1_probe).mean(-1)
                all_num += sent_cos_neg.size()[0]
                pos = sent_cos_pos + 1
                pos_num += torch.nonzero(pos>1).size()[0]
                neg = sent_cos_neg + 1
                neg_num += torch.nonzero(neg<1).size()[0]
                loss = F.relu(self.margin - sent_cos_pos + sent_cos_neg)
                loss_all.append(loss.mean().item())

            # loss_all.append(F.relu(self.margin - sent_wmd_neg + sent_wmd_pos).mean().item())
        pos_accuracy = pos_num/all_num
        neg_accuracy = neg_num/all_num
        all_accuracy = (pos_num + neg_num)/(2*all_num)
        return np.mean(loss_all), (pos_accuracy, neg_accuracy, all_accuracy)

