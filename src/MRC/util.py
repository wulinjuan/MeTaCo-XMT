from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random
from transformers import BertTokenizer, XLMRobertaTokenizer, XLMTokenizer

logger = logging.getLogger(__name__)

langs = {
    'xquad': ['zh', 'en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi'],  # Exclude 'th'
    'mlqa': ['en', 'ar', 'de', 'es', 'hi', 'vi', 'zh'],
    'tydiqa': ['en', 'ar', 'bg', 'fi', 'id', 'ko', 'ru', 'sw', 'te'] # 'en', 'ar', 'bg', 'fi', 'id', 'ko', 'ru', 'sw', 'te'
}


def flatten(l):
    return [item for sublist in l for item in sublist]


def initialize_config(config_name, sub_name, config_file_path,create_dir=True):
    logger.info("Experiment: {}".format(config_name))

    config = pyhocon.ConfigFactory.parse_file(config_file_path)[config_name]

    config['log_dir'] = join(config["log_root"], config_name+sub_name)
    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    if create_dir:
        makedirs(config['log_dir'], exist_ok=True)
        makedirs(config['tb_dir'], exist_ok=True)

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def get_bert_tokenizer(config):
    # Avoid using fast tokenization
    if config['model_type'] == 'xlm':
        return XLMTokenizer.from_pretrained(config['pretrained'])
    elif 'bert' in config['model_type']:
        return BertTokenizer.from_pretrained(config['pretrained'])
    elif 'xlmr' in config['model_type'] or "infoxlm" in config['model_type']:
        return XLMRobertaTokenizer.from_pretrained(config['pretrained'])
    else:
        raise ValueError('Unknown model type')
