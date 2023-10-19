# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/squad.py

import json
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool, cpu_count
import logging
from os.path import join

import numpy
import numpy as np
from torch import nn
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from transformers import BertForQuestionAnswering, XLMRobertaForQuestionAnswering, XLMForQuestionAnswering
from transformers.tokenization_utils_base import TruncationStrategy

from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm.modeling_xlm import XLMModel

logger = logging.getLogger(__name__)

def get_seq_encoder(config):
    if 'bert' in config['model_type']:
        return BertModel.from_pretrained(config['pretrained'])
    elif 'xlmr' in config['model_type'] or "infoxlm" in config['model_type']:
        return XLMRobertaModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlm':
        return XLMModel.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"xlmroberta", "roberta", "camembert", "bart"}  # Why roberta??

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    # e.g. orig_answer_text = '***' but last original token is '***.'
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token; a token can appear in multiple overlapping segments"""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def get_QA_model(config):
    if config['model_type'] == 'bert':
        return BertForQuestionAnswering.from_pretrained(config['pretrained'])
    elif 'xlmr' in config['model_type'] or "infoxlm" in config['model_type']:
        return XLMRobertaForQuestionAnswering.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlm':
        return XLMForQuestionAnswering.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])

def compute_represenation(sents, config, model_path=None):
    device = f'cuda:{config.gpu_id}'
    if model_path:
        # model = TransformerQa(config)
        model = get_QA_model(config)
        file_name = model_path.split('_seed')[0]
        model_path = join(f'../../data/{file_name}/',
                         f'model_{model_path}/pytorch_model.bin')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        if config['model_type'] == 'bert':
            model = model.bert
        elif 'xlmr' in config['model_type'] or "infoxlm" in config['model_type']:
            model = model.roberta
        elif config['model_type'] == 'xlm':
            model = model.transformer
    else:
        model = get_seq_encoder(config)
    model.eval()
    model.to(device)
    batch_size = 16
    for i in range(0, len(sents), batch_size):
        items = sents[i: min(len(sents), i + batch_size)]
        with torch.no_grad():
            input_ids = torch.tensor([item.input_ids for item in items], dtype=torch.long).to(device)
            segment_ids = torch.tensor([item.token_type_ids for item in items], dtype=torch.long).to(device)
            input_mask = torch.tensor([item.attention_mask for item in items], dtype=torch.long).to(device)
            if model_path is None:
                all_encoder_layers = model(input_ids, input_mask, segment_ids, output_attentions=True, output_hidden_states=False, return_dict=True)
            else:
                all_encoder_layers = model(input_ids, input_mask, segment_ids)
        try:
            # layer_output = all_encoder_layers[1].detach().cpu().numpy()  # batch_size x target_size
            layer_output = all_encoder_layers['pooler_output'].detach().cpu().numpy()
            input_rep = all_encoder_layers['last_hidden_state'].detach().cpu().numpy()
            attention = all_encoder_layers['attentions'][-1]
            attention = torch.index_select(attention, 2, torch.tensor([0]).to('cuda:{}'.format(config.gpu_id))).squeeze(2)
            attention = torch.mean(attention, dim=1).detach().cpu().numpy()
        except:
            attention = None
            input_rep = None
            layer_output = all_encoder_layers[0].detach().cpu().mean(axis=1, keepdim=False).numpy()
        for j, item in enumerate(items):
            item.representation = layer_output[j]
            if attention is not None:
                item.attention = attention[j]
                item.input_rep = input_rep[j]
        # item.representation = layer_output
        if i % (1000 * batch_size) == 0:
            logger.info('  Compute sentence representation. To {}...'.format(i))
    logger.info('  Finish.')
    model.to('cpu')
    del model


def squad_convert_example_to_features(
        example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # Check sanitization; if the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(example.answer_text.strip().split())
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    # Build map between subtokens and original tokens
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # Get answer positions for subtokens
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    # Encode entire doc into multiple segments with stride
    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        # Encode sentence-pair
        encoded_dict = tokenizer.encode_plus(
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            # Overlapping/encoding length
            return_token_type_ids=True,
        )

        paragraph_len = min(  # Length of the context just encoded
            len(all_doc_tokens) - len(spans) * doc_stride,  # Actual remaining length
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,  # Max allowable length
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                    tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}  # Current context subtok index to original tok
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len  # Length of context
        encoded_dict["tokens"] = tokens  # Encoded sentence-pair without padding
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride  # Start idx in entire doc subtoks
        encoded_dict["length"] = paragraph_len

        # Sanity check
        assert encoded_dict["input_ids"][len(truncated_query) + sequence_added_tokens] == \
               tokenizer.convert_tokens_to_ids(all_doc_tokens[encoded_dict["start"]])

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    # Fill token_is_max_context
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id) if config['model_type'] != 'mt5' else -1

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        # Excluded tokens: padding, [SEP], query
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        # If the example always have answers, filter out spans without gold answers
        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],  # Full encoded sentence-pair
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],  # Length of current context
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],  # Encoded sentence-pair without padding
                token_to_orig_map=span["token_to_orig_map"],  # Context subtok index to original tok
                start_position=start_position,  # Start idx of answer in current encoded
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert, config_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

    global config
    config = config_for_convert

def squad_convert_examples_to_meta_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        config,
        query_feature_file,
        padding_strategy="max_length",
        return_dataset=False,
        threads=4,
        tqdm_enabled=True,
        get_meta_data=False,
        train_features=None,
        cl_model=None,
        model_path=None
):
    # Convert each example to multiple input features
    if query_feature_file is None or not os.path.exists(query_feature_file):
        threads = min(threads, cpu_count())
        with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer, config)) as p:
            annotate_ = partial(
                squad_convert_example_to_features,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                padding_strategy=padding_strategy,
                is_training=is_training,
            )
            qry_features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert squad examples to features",
                    disable=not tqdm_enabled,
                )
            )

        # Add example idx and unique id
        new_features = []
        unique_id = 1000000000
        example_index = 0

        for example_features in tqdm(
                qry_features, total=len(qry_features), desc="add example index and unique id", disable=not tqdm_enabled
        ):
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = unique_id
                new_features.append(example_feature)
                unique_id += 1
            example_index += 1
        qry_features = new_features
        del new_features
        if is_training:
            compute_represenation(sents=qry_features, config=config, model_path=model_path)

        if get_meta_data:
            if train_features:
                qry_features = build_support_features_(train_features, qry_features, support_size=10, cl_model=cl_model)
            else:
                qry_features = build_support_features_(qry_features, support_size=10, cl_model=cl_model)
        if query_feature_file is not None:
            with open(query_feature_file, 'wb') as f:
                pickle.dump(qry_features, f, protocol=4)
    else:
        with open(query_feature_file, 'rb') as f:
            data = pickle.load(f)
            if len(data) == 3:
                _, qry_features, _ = data
            else:
                qry_features = data
    spt_features = train_features
    del train_features
    l_input_ids_s = []
    l_attention_masks_s = []
    l_token_type_ids_s = []
    l_cls_index_s = []
    l_p_mask_s = []
    l_example_index_s = []
    l_start_positions_s = []
    l_end_positions_s = []

    l_input_ids_q = []
    l_feature_index = []
    l_attention_masks_q = []
    l_token_type_ids_q = []
    l_cls_index_q = []
    l_p_mask_q = []
    l_example_index_q = []
    l_start_positions_q = []
    l_end_positions_q = []

    if is_training:
        #random.seed(111)
        random.shuffle(qry_features)
    else:
        config["batch_sz"] = len(qry_features)
    # 出现频率过大的样本，限制在一定的概率出现
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

    # 出现频率过大的样本，限制在一定的概率出现
    skip_num = []
    for dic in dic_num:
        if dic[1] * 2 > len(dic_num):
            skip_num.append(dic[0])
        else:
            break

    s = 0
    print("META Batching")
    for _ in tqdm(range(config["batch_sz"])):
        l_input_ids_spt = []
        l_attention_masks_spt = []
        l_token_type_ids_spt = []
        l_all_cls_index_spt = []
        l_p_mask_spt = []
        l_start_positions_spt = []
        l_end_positions_spt = []

        l_input_ids_qry = []
        l_attention_masks_qry = []
        l_token_type_ids_qry = []
        l_all_cls_index_qry = []
        l_p_mask_qry = []
        l_start_positions_qry = []
        l_end_positions_qry = []

        # Pick q_qry from qry examples randomly
        if s + config["q_qry"] < len(qry_features):
            qry_indices = range(s, s + config["q_qry"])
            s = s + config["q_qry"]
        elif s < len(qry_features):
            t = s + config["q_qry"] - len(qry_features)
            qry_indices = list(range(s, len(qry_features))) + list(range(0, t))
            s = t
        else:
            s = 0
            qry_indices = range(s, len(qry_features))

        all_feature_index = torch.tensor(qry_indices, dtype=torch.long)
        l_feature_index.append(all_feature_index)

        # random.sample(len(qry_examples), k=data_config["q_qry"])

        for index in qry_indices:
            # Pick qry features
            l_input_ids_qry.append(qry_features[index].input_ids)
            l_attention_masks_qry.append(qry_features[index].attention_mask)
            l_token_type_ids_qry.append(qry_features[index].token_type_ids)
            l_all_cls_index_qry.append(qry_features[index].cls_index)
            l_p_mask_qry.append(qry_features[index].p_mask)
            l_start_positions_qry.append(qry_features[index].start_position)
            l_end_positions_qry.append(qry_features[index].end_position)

            # Pick s_spt from spt examples based on rankings
            if skip_num == []:
                spt_indices = qry_features[index].rankings_spt[:config['meta_example_num']]
            else:
                spt_indices = []
                i = 0
                while len(spt_indices) < config['meta_example_num']:
                    if qry_features[index].rankings_spt[i] not in skip_num:
                        spt_indices.append(qry_features[index].rankings_spt[i])
                    i += 1

            for spt_index in spt_indices:
                l_input_ids_spt.append(spt_features[spt_index].input_ids)
                l_attention_masks_spt.append(spt_features[spt_index].attention_mask)
                l_token_type_ids_spt.append(spt_features[spt_index].token_type_ids)
                l_all_cls_index_spt.append(spt_features[spt_index].cls_index)
                l_p_mask_spt.append(spt_features[spt_index].p_mask)
                l_start_positions_spt.append(spt_features[index].start_position)
                l_end_positions_spt.append(spt_features[index].end_position)

        all_input_ids = torch.tensor(l_input_ids_spt, dtype=torch.long)
        l_input_ids_s.append(all_input_ids)
        l_attention_masks_s.append(torch.tensor(l_attention_masks_spt, dtype=torch.long))
        l_token_type_ids_s.append(torch.tensor(l_token_type_ids_spt, dtype=torch.long))
        l_cls_index_s.append(torch.tensor(l_all_cls_index_spt, dtype=torch.long))
        l_p_mask_s.append(torch.tensor(l_p_mask_spt, dtype=torch.float))
        l_example_index_s.append(torch.arange(all_input_ids.size(0), dtype=torch.long))
        l_start_positions_s.append(torch.tensor(l_start_positions_spt, dtype=torch.long))
        l_end_positions_s.append(torch.tensor(l_end_positions_spt, dtype=torch.long))

        all_input_ids = torch.tensor(l_input_ids_qry, dtype=torch.long)
        l_input_ids_q.append(all_input_ids)
        l_attention_masks_q.append(torch.tensor(l_attention_masks_qry, dtype=torch.long))
        l_token_type_ids_q.append(torch.tensor(l_token_type_ids_qry, dtype=torch.long))
        l_cls_index_q.append(torch.tensor(l_all_cls_index_qry, dtype=torch.long))
        l_p_mask_q.append(torch.tensor(l_p_mask_qry, dtype=torch.float))
        l_example_index_q.append(torch.arange(all_input_ids.size(0), dtype=torch.long))
        l_start_positions_q.append(torch.tensor(l_start_positions_qry, dtype=torch.long))
        l_end_positions_q.append(torch.tensor(l_end_positions_qry, dtype=torch.long))

    if return_dataset:
        if is_training:
            dataset = TensorDataset(
                torch.stack(l_input_ids_q),
                torch.stack(l_attention_masks_q),
                torch.stack(l_token_type_ids_q), torch.stack(l_start_positions_q), torch.stack(l_end_positions_q),
                torch.stack(l_cls_index_q), torch.stack(l_p_mask_q),
                torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s),
                torch.stack(l_start_positions_s), torch.stack(l_end_positions_s), torch.stack(l_cls_index_s),
                torch.stack(l_p_mask_s))
        else:
            dataset = TensorDataset(
                torch.stack(l_input_ids_q),
                torch.stack(l_attention_masks_q),
                torch.stack(l_token_type_ids_q),
                torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s), torch.stack(l_token_type_ids_s),
                torch.stack(l_start_positions_s), torch.stack(l_end_positions_s), torch.stack(l_cls_index_s),
                torch.stack(l_p_mask_s), torch.stack(l_feature_index))
        return spt_features, qry_features, dataset
    return spt_features, qry_features


def squad_convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        config,
        padding_strategy="max_length",
        return_dataset=False,
        threads=4,
        tqdm_enabled=True,
        get_meta_data=False,
        train_features=None,
        cl_model=None,
        cand_features_file=None,
        model_path=None,
):
    # Convert each example to multiple input features
    features = []
    if len(examples) == 2:
        examples = examples[0]
    if cand_features_file is not None and os.path.exists(cand_features_file):
        with open(cand_features_file, 'rb') as f:
            features=pickle.load(f)
    else:
        threads = min(threads, cpu_count())
        with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer, config)) as p:
            annotate_ = partial(
                squad_convert_example_to_features,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                padding_strategy=padding_strategy,
                is_training=is_training,
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert squad examples to features",
                    disable=not tqdm_enabled,
                )
            )

        # Add example idx and unique id
        new_features = []
        unique_id = 1000000000
        example_index = 0

        for example_features in tqdm(
                features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
        ):
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = unique_id
                new_features.append(example_feature)
                unique_id += 1
            example_index += 1
        features = new_features
        del new_features
        if is_training:
            compute_represenation(sents=features, config=config, model_path=model_path)

        if get_meta_data:
            if train_features:
                features = build_support_features_(train_features, features, support_size=config.meta_example_num,
                                                   cl_model=cl_model)
            else:
                features = build_support_features_(features, support_size=config.meta_example_num, cl_model=cl_model)

        if cand_features_file is not None:
            with open(cand_features_file, 'wb') as f:
                pickle.dump(features, f, protocol=4)

    if len(features) == 2:
        features = features[1]

    # Return dataset
    if return_dataset:
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([[f.input_ids] for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([[f.attention_mask] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([[f.token_type_ids] for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        # representation = torch.tensor([numpy.array(f.representation) for f in features], dtype=torch.float)
        if get_meta_data:
            if train_features:
                feature_base = train_features
            else:
                feature_base = features
            meta_data_input_ids = []
            meta_data_attention_masks = []
            meta_data_token_type_ids = []
            meta_start_positions = []
            meta_end_positions = []
            for f in features:
                meta_data_input_ids1 = []
                meta_data_attention_masks1 = []
                meta_data_token_type_ids1 = []
                meta_start_positions1 = []
                meta_end_positions1 = []
                for idx in f.meta_set_num:
                    meta_data_input_ids1.append(feature_base[idx].input_ids)
                    id = feature_base[idx].qas_id
                    meta_data_attention_masks1.append(feature_base[idx].attention_mask)
                    meta_data_token_type_ids1.append(feature_base[idx].token_type_ids)
                    meta_start_positions1.append(feature_base[idx].start_position)
                    meta_end_positions1.append(feature_base[idx].end_position)
                meta_data_input_ids.append(meta_data_input_ids1)
                meta_data_attention_masks.append(meta_data_attention_masks1)
                meta_data_token_type_ids.append(meta_data_token_type_ids1)
                meta_start_positions.append(meta_start_positions1)
                meta_end_positions.append(meta_end_positions1)
            meta_data_input_ids = torch.tensor(meta_data_input_ids, dtype=torch.long)
            meta_data_attention_masks = torch.tensor(meta_data_attention_masks, dtype=torch.long)
            meta_data_token_type_ids = torch.tensor(meta_data_token_type_ids, dtype=torch.long)
            meta_start_positions = torch.tensor(meta_start_positions, dtype=torch.long)
            meta_end_positions = torch.tensor(meta_end_positions, dtype=torch.long)

        if not is_training and train_features==None:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)  # To identify feature in batch
            if get_meta_data:
                dataset = TensorDataset(
                    all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask,
                    meta_data_input_ids,
                    meta_data_attention_masks,
                    meta_data_token_type_ids,
                    meta_start_positions,
                    meta_end_positions,
                )
            else:
                dataset = TensorDataset(
                    all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index,
                    all_p_mask
                )
        elif not is_training and get_meta_data and train_features:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)  # To identify feature in batch
            all_start_positions = torch.tensor([[0] for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([[0] for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_feature_index,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                meta_data_input_ids,
                meta_data_attention_masks,
                meta_data_token_type_ids,
                meta_start_positions,
                meta_end_positions,
                all_is_impossible,
            )
        else:
            all_start_positions = torch.tensor([[f.start_position] for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([[f.end_position] for f in features], dtype=torch.long)
            if get_meta_data:
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_start_positions,
                    all_end_positions,
                    all_cls_index,
                    all_p_mask,
                    meta_data_input_ids,
                    meta_data_attention_masks,
                    meta_data_token_type_ids,
                    meta_start_positions,
                    meta_end_positions,
                    all_is_impossible,
                )
            else:
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_start_positions,
                    all_end_positions,
                    all_cls_index,
                    all_p_mask,
                    all_is_impossible,
                )

        return features, dataset
    else:
        return features


def build_support_features_1(base_features, target_features=None, support_size=2, cl_mfodel=None):
    if target_features == None:
        target_features = base_features
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
    for i, item in enumerate(target_features):
        chosen_ids = relevance[i][-1 * (support_size + 1): -1]
        # logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
        # support = [base_features[id] for id in chosen_ids]
        # support_set.append(support)
        if cl_model is not None:
            chosen_dict = {}
            target_features[i].meta_set_num=[]
            target_representation = torch.tensor([target_features[i].input_rep], dtype=torch.float).to(cl_model.device)
            target_attention = torch.tensor([target_features[i].attention], dtype=torch.float).to(cl_model.device)
            target_latent_representation = torch.matmul(target_representation, cl_model.probe.proj)
            for id in chosen_ids:
                chosen_representation = torch.tensor([base_features[id].input_rep], dtype=torch.float).to(cl_model.device)
                chosen_attention = torch.tensor([base_features[i].attention], dtype=torch.float).to(cl_model.device)
                chosen_latent_representation = torch.matmul(chosen_representation, cl_model.probe.proj)
                score = cl_model.get_wmd(target_attention, chosen_attention, target_latent_representation, chosen_latent_representation).mean().item()
                if id not in chosen_dict:
                    chosen_dict[id] = score
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


def build_support_features_(base_features, target_features=None, support_size=2, cl_model=None):
    if target_features == None:
        target_features = base_features
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
            target_features[i].meta_set_num=[]
            target_representation = torch.tensor([target_features[i].input_rep], dtype=torch.float).repeat(support_size, 1, 1).to(cl_model.device)
            target_attention = torch.tensor([target_features[i].attention], dtype=torch.float).repeat(support_size, 1).to(cl_model.device)
            target_latent_representation = torch.matmul(target_representation, cl_model.probe.proj)

            chosen_representation = torch.tensor([base_features[id].input_rep for id in chosen_ids], dtype=torch.float).to(cl_model.device)
            chosen_attention = torch.tensor([base_features[id].attention for id in chosen_ids], dtype=torch.float).to(cl_model.device)
            chosen_latent_representation = torch.matmul(chosen_representation, cl_model.probe.proj)
            score = cl_model.get_wmd(target_attention, chosen_attention, target_latent_representation, chosen_latent_representation).tolist()
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


class SquadProcessor:
    """ Process squad-like dataset """

    def get_train_examples(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            input_data = json.load(f)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_or_test_examples(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            input_data = json.load(f)["data"]
        return self._create_examples(input_data, "dev.txt")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry.get("title", "none")
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers=[],
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0  # Inclusive

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens  # Cleaned tokens split on whitespace
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class SquadFeatures:
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            start_position,
            end_position,
            is_impossible,
            representation=None,
            input_rep=None,
            attention=None,
            rankings_spt=None,
            qas_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id
        self.representation = representation
        self.input_rep = input_rep
        self.attention = attention
        self.rankings_spt = rankings_spt


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits