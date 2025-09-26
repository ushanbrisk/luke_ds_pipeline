import torch
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np

class DataCollatorForPromptDataset(object):
    """Collate for supervised fine-tuning."""
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, samples):
        #print(f" process: {os.getpid()}")
        input_ids_list, labels_list = [], []

        # bz = len(samples)
        # for idx in range(bz):
        #     if len(samples[idx]["input_ids"]) == self.max_len:
        #         break
        #     else:
        #         samples[idx]["input_ids"] = samples[idx]["input_ids"] + \
        #             [self.tokenizer.pad_token_id] * (self.max_len -len(samples[idx]["input_ids"]))
        #

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, samples, padding='max_length',return_tensors="pt",max_length=self.max_len
        )

        # for instance in batch:
        #     input_ids = torch.tensor(instance["input_ids"], dtype=torch.int64)
        #     # input_ids = instance["input_ids"]
        #     input_ids_list.append(input_ids)
        #     # labels_list.append(instance["labels"])
        #     labels = input_ids.clone()  #copy from DataCollatorForLanguageModeling
        #     if self.tokenizer.pad_token_id is not None:
        #         labels[labels==self.tokenizer.pad_token_id] = -100
        #     labels_list.append(labels)

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        # return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))
        # return ((batch['input_ids'], batch['labels']), batch['labels'])

        return ((batch['input_ids'], batch['labels']), torch.tensor([43]))
        # return ((batch['input_ids'], torch.tensor([43])), batch['labels'])
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded



class DataCollatorForPromptDatasetDummy(object):
    """Collate for supervised fine-tuning."""
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, samples):
        #print(f" process: {os.getpid()}")
        input_ids_list, labels_list = [], []

        # bz = len(samples)
        # for idx in range(bz):
        #     if len(samples[idx]["input_ids"]) == self.max_len:
        #         break
        #     else:
        #         samples[idx]["input_ids"] = samples[idx]["input_ids"] + \
        #             [self.tokenizer.pad_token_id] * (self.max_len -len(samples[idx]["input_ids"]))
        #

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, samples, padding='max_length',return_tensors="pt",max_length=self.max_len
        )

        # for instance in batch:
        #     input_ids = torch.tensor(instance["input_ids"], dtype=torch.int64)
        #     # input_ids = instance["input_ids"]
        #     input_ids_list.append(input_ids)
        #     # labels_list.append(instance["labels"])
        #     labels = input_ids.clone()  #copy from DataCollatorForLanguageModeling
        #     if self.tokenizer.pad_token_id is not None:
        #         labels[labels==self.tokenizer.pad_token_id] = -100
        #     labels_list.append(labels)

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        # return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))
        # return ((batch['input_ids'], batch['labels']), batch['labels'])

        # return ((batch['input_ids'], batch['labels']), torch.tensor([43]))
        return ((batch['input_ids'], torch.tensor([43])), batch['labels'])

@dataclass
class DataCollatorMedical:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                            (max_label_length + self.pad_to_multiple_of - 1)
                            // self.pad_to_multiple_of
                            * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)),
                                         dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)),
                                         dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

#        return batch    #remove
        return ((batch['input_ids'], torch.tensor([43])), batch['labels'])


    
    