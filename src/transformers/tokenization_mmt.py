import json
import re
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece

from .file_utils import add_start_docstrings
from .tokenization_utils import BatchEncoding
from .tokenization_marian import MarianTokenizer
from .tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING


vocab_files_names = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "source_spm": {"Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/source.spm"},
    "target_spm": {"Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/target.spm"},
    "vocab": {"Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/vocab.json"},
    "tokenizer_config_file": {
        "Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/tokenizer_config.json"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"Helsinki-NLP/opus-mt-en-de": 512}
PRETRAINED_INIT_CONFIGURATION = {}

# Example URL https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-en-de/vocab.json


class MMTTokenizer(MarianTokenizer):
    r"""
    Construct a Marian tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        source_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the source language.
        target_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the target language.
        source_lang (:obj:`str`, `optional`):
            A string representing the source language.
        target_lang (:obj:`str`, `optional`):
            A string representing the target language.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (:obj:`int`, `optional`, defaults to 512):
            The maximum sentence length the model accepts.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.

    Examples::

        >>> from transformers import MarianTokenizer
        >>> tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        >>> src_texts = [ "I am a small frog.", "Tom asked his teacher for advice."]
        >>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
        >>> batch_enc: BatchEncoding = tok.prepare_seq2seq_batch(src_texts, tgt_texts=tgt_texts)
        >>> # keys  [input_ids, attention_mask, labels].
        >>> # model(**batch) should work
    """

    vocab_files_names = vocab_files_names
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]
    language_code_re = re.compile(">>.+<<")  # type: re.Pattern

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        return_tensors: str = "pt",
        truncation=True,
        padding="longest",
        **unused,
    ) -> BatchEncoding:
        if "" in src_texts:
            raise ValueError(f"found empty string in src_texts: {src_texts}")
        self.current_spm = self.spm_source
        src_texts = [self.normalize(t) for t in src_texts]  # this does not appear to do much
        tokenizer_kwargs = dict(
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        model_inputs: BatchEncoding = self(src_texts, **tokenizer_kwargs)

        if tgt_texts is None:
            return model_inputs
        if max_target_length is not None:
            tokenizer_kwargs["max_length"] = max_target_length

        self.current_spm = self.spm_target
        model_inputs["labels"] = self(tgt_texts, **tokenizer_kwargs)["input_ids"]
        self.current_spm = self.spm_source
        return model_inputs

    def num_special_tokens_to_add(self, **unused):
        """Just EOS"""
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]
