import os
import hashlib

import torch
import torch.nn as nn

import numpy as np

# ELMO Dependencies. TODO: Remove.
from allennlp.commands.elmo import ElmoEmbedder

from tqdm import tqdm

from .configuration_diora import DioraConfig
from .modeling_utils import PretrainedModel
from .utils import get_logger


DIORA_PRETRAINED_MODEL_ARCHIVE_MAP = {}


logger = get_logger()


class DioraPretrainedModel(PretrainedModel):
    config_class = DioraConfig
    pretrained_model_archive_map = DIORA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "diora"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DioraEmbeddings(nn.Module):
    version = "v0.0.0"

    def __init__(self, config, vocab_to_cache):
        super().__init__()

        self.config = config
        self.cache_embeddings(vocab_to_cache)

    def load_cached_embeddings(self, path):
        return np.load(path)

    def save_cached_embeddings(self, path, vectors):
        np.save(path, vectors)

    def validate_vocab(self, vocab_to_cache):
        assert isinstance(vocab_to_cache, dict)
        sorted_vocab = [w for w, i in sorted(vocab_to_cache.items(), key=lambda x: x[1])]
        for i, w in enumerate(sorted_vocab):
            assert vocab_to_cache[w] == i

    def hash_vocab(self, vocab_to_cache):
        sorted_vocab = [w for w, i in sorted(vocab_to_cache.items(), key=lambda x: x[1])]
        m = hashlib.sha256()
        m.update(str.encode(self.version))
        for w in sorted_vocab:
            m.update(str.encode(w))
        return m.hexdigest()

    def cache_embeddings(self, vocab_to_cache):
        cache_dir = os.path.expanduser("~/.parsing/cached_embeddings")
        assert os.path.exists(cache_dir), "Create the cached embeddings directory."

        self.validate_vocab(vocab_to_cache)
        key = self.hash_vocab(vocab_to_cache)
        cache_path = os.path.join(cache_dir, key) + ".npy"

        logger.info("Cache path = {}".format(cache_path))

        if os.path.exists(cache_path):
            logger.info("Loading cached elmo vectors: {}".format(cache_path))
            vectors = self.load_cached_embeddings(cache_path)
            logger.info("Loaded with shape = {}".format(vectors.shape))
            return vectors

        device = 0 if self.config.cuda else -1
        logger.info("Embeddings not cached. Initializing elmo.")
        elmo = ElmoEmbedder(
            options_file=self.config.elmo_options_path, weight_file=self.config.elmo_weights_path, cuda_device=device
        )

        batch_size = 256
        nbatches = len(vocab_to_cache) // batch_size + 1
        vocab = [w for w, i in sorted(vocab_to_cache.items(), key=lambda x: x[1])]

        logger.info("Computing context-insensitive character embeddings. vocab-size = {}".format(len(vocab)))
        with torch.no_grad():
            vec_lst = []
            for i in tqdm(range(nbatches), desc="elmo"):
                start = i * batch_size
                batch = vocab[start : start + batch_size]
                if len(batch) == 0:
                    continue
                vec = elmo.embed_sentence(batch)
                vec_lst.append(vec)

        vectors = np.concatenate([x[0] for x in vec_lst], axis=0)
        logger.info("Computed with shape = {}".format(vectors.shape))

        self.save_cached_embeddings(cache_path, vectors)

        return vectors


class DioraEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()


class DioraModel(DioraPretrainedModel):
    def __init__(self, config, vocab_to_cache):
        super().__init__(config)
        self.config = config

        self.embeddings = DioraEmbeddings(config, vocab_to_cache)
        self.encoder = DioraEncoder(config)

        self.init_weights()
