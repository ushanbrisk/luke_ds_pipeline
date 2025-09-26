# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
from dschat.utils.data import raw_datasets
from deepspeed.accelerator import get_accelerator

from dschat.utils.data.raw_datasets import maybe_apply_chat_template


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "ricdomolm/MATH-500" in dataset_name:
        return raw_datasets.RicdomolmMath500Dataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "lukedai/test" in dataset_name:
        return raw_datasets.LukedaiTestDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "open-r1/OpenR1-Math-220k" in dataset_name:
        return raw_datasets.Openr1Openr1math220kDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "lukedai/delicate_medical_r1_data_eval" in dataset_name:
        return raw_datasets.MedicalR1Dataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    elif "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir,
                         os.path.pardir, os.path.pardir))
        if not (os.path.isfile(chat_path + '/data/train.json')
                and os.path.isfile(chat_path + '/data/eval.json')):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank,
                                                 dataset_name, chat_path)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank,
                                output_path,
                                dataset_name,
                                seed,
                                split_name,
                                data_split,
                                split_index,
                                data_size,
                                rebuild=False):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    # reindex each time when using local jsonfile since it's more likely to get modified
    if rebuild or (not os.path.isfile(index_file_name)) or (dataset_name
                                                            == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids":
                self.chosen_dataset[idx]["input_ids"],
                "attention_mask":
                self.chosen_dataset[idx]["attention_mask"],
                "labels":  #needs to consider, when packing, input_ids is less than max_seq_len, and labels the same; but after collator, input_ids is padded, but label not
                torch.where(self.chosen_dataset[idx]["attention_mask"].bool(),
                            self.chosen_dataset[idx]["input_ids"], -100)
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data, tokenizer)  # the accept response
            if chosen_sentence is not None:
                if end_of_conversation_token not in chosen_sentence:
                    chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 3:
        filtered = 0
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                if prompt_token["input_ids"].size()[-1] <= max_seq_len:
                    for key_word in ["input_ids", "attention_mask"]:
                        prompt_token[key_word] = prompt_token[
                            key_word].squeeze(0).flip(0)
                    prompt_dataset.append(prompt_token)
                else:
                    filtered += 1
        print(f'Creating dataset {raw_dataset.dataset_name_clean} '
              f'for {train_phase=} size={len(prompt_dataset)} {filtered=}')

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, rebuild):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset), rebuild)
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset), rebuild)
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print(f'Creating prompt dataset {data_path}, {reload=}')
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank,
                data_path[0],
                data_split,
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank,
                    d_path,
                    data_split,
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    rebuild=reload)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    rebuild=reload)
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    torch.distributed.barrier()
    return torch.load(train_fname, weights_only=False), torch.load(eval_fname,weights_only=False)  #modifed 20250402 , weights_only=true
    # return torch.load(train_fname), torch.load(eval_fname)  #modifed 20250402 , weights_only=true


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []


#
# #????train_phase=1???
# def create_dataset_2(local_rank, dataset_name, data_split, output_path,
#                    train_phase, seed, tokenizer, end_of_conversation_token,
#                    max_seq_len, rebuild):
#     raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
#
#     #train dataset
#     train_dataset = raw_dataset.get_train_data()
#     map_kwargs = {}
#     map_kwargs["num_proc"] = 10 #here is the parallel process number
#     map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
#     train_dataset = train_dataset.map(maybe_apply_chat_template,fn_kwargs={"tokenizer": tokenizer, "is_output_dict":True},**map_kwargs)
#     map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
#     train_dataset = train_dataset.map(lambda ex: tokenizer(ex["text"]), **map_kwargs)
#     train_index = get_raw_dataset_split_index(local_rank, output_path,
#                                               raw_dataset.dataset_name_clean,
#                                               seed, "train", data_split,
#                                               train_phase - 1,
#                                               len(train_dataset), rebuild)
#     train_dataset = Subset(train_dataset, train_index)
#     prompt_dataset = []
#     chosen_dataset = []
#     reject_dataset = []
#     for i, tmp_data in enumerate(train_dataset):#??current_dataset?????
#         chosen_dataset.append({'input_ids':torch.Tensor(tmp_data['input_ids']), 'attention_mask':torch.Tensor(tmp_data['attention_mask'])})
#     train_dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
#                          tokenizer.pad_token_id, train_phase)
#
#     #eval dataset
#     eval_dataset = raw_dataset.get_eval_data()
#     map_kwargs = {}
#     map_kwargs["num_proc"] = 10 #here is the parallel process number
#     map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
#     eval_dataset = eval_dataset.map(maybe_apply_chat_template,fn_kwargs={"tokenizer": tokenizer, "is_output_dict":True},**map_kwargs)
#     map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
#     eval_dataset = eval_dataset.map(lambda ex: tokenizer(ex["text"]), **map_kwargs)
#     eval_index = get_raw_dataset_split_index(local_rank, output_path,
#                                               raw_dataset.dataset_name_clean,
#                                               seed, "eval", data_split,
#                                               train_phase - 1,
#                                               len(eval_dataset), rebuild)
#
#     eval_dataset = Subset(eval_dataset, eval_index)
#     prompt_dataset = []
#     chosen_dataset = []
#     reject_dataset = []
#     for i, tmp_data in enumerate(eval_dataset):#??current_dataset?????
#         chosen_dataset.append({'input_ids':torch.Tensor(tmp_data['input_ids']), 'attention_mask':torch.Tensor(tmp_data['attention_mask'])})
#     eval_dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
#                          tokenizer.pad_token_id, train_phase)
#     return train_dataset, eval_dataset
#

def create_prompt_dataset_2(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False):
    """
    Creates the prompt dataset
    """

    ################
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    #depends on tokenizer, as need tokenzier text
    fname = f"only_train_full_{fname}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"

    cache_found = os.path.isfile(train_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)
    ####

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print(f'Creating prompt dataset {data_path}, {reload=}')

        if len(data_path) == 1:  # Single dataset.
            train_dataset = create_dataset_2(
                local_rank,
                data_path[0],
                data_split,
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload)
        print(f'finish read dataset')
        torch.save(train_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname, weights_only=False) # modifed 20250402 , weights_only=true

#????train_phase=1???
def create_dataset_2(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, rebuild):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    print("finished get raw_data\n")
    #train dataset
    train_dataset = raw_dataset.get_train_data()
    train_dataset = train_dataset.select_columns(['messages'])
    del raw_dataset
    print("finished selecting message column")

    map_kwargs = {}
    map_kwargs["num_proc"] = 52 #here is the parallel process number
    map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
    train_dataset = train_dataset.map(maybe_apply_chat_template,fn_kwargs={"tokenizer": tokenizer, "is_output_dict":True},**map_kwargs)
    print("finish maybe_apply_chat_template() to all data\n")
    train_dataset = train_dataset.select_columns(['text'])
    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
    map_kwargs["num_proc"] = 52
    train_dataset = train_dataset.map(lambda ex: tokenizer(ex["text"]), **map_kwargs)
    print("finish tokenizering all data\n")

    train_dataset = train_dataset.select_columns("input_ids")
    map_kwargs["desc"] = f"Packing {dataset_name} dataset"
    # import os
    # print(f"os pid: {os.getpid()}")
    train_dataset = train_dataset.map(
        pack_examples, batched=True, fn_kwargs={"seq_length": max_seq_len}, **map_kwargs
    )
    # train_dataset["input_ids"]=torch.Tensor(train_dataset["input_ids"])
    #remove this split index process
    # train_index = get_raw_dataset_split_index(local_rank, output_path,
    #                                           raw_dataset.dataset_name_clean,
    #                                           seed, "train", data_split,
    #                                           train_phase - 1,
    #                                           len(train_dataset), rebuild)
    # print("finish creating train index\n")
    # train_dataset = Subset(train_dataset, train_index)
    # prompt_dataset = []
    # chosen_dataset = []
    # reject_dataset = []
    # for i, tmp_data in enumerate(train_dataset):#??current_dataset????? the purpose of this action is fetch data from random selection order
    #     # chosen_dataset.append({'input_ids':torch.Tensor(tmp_data['input_ids']), 'attention_mask':torch.Tensor(tmp_data['attention_mask'])})
    #     # chosen_dataset.append(tmp_data['input_ids'])
    #     chosen_dataset.append({'input_ids':tmp_data['input_ids'], 'attention_mask':tmp_data['attention_mask']})
    #
    #
    # print("finish gather all [dict()] data to chosen_dataset\n" )
    # from datasets.arrow_dataset import Dataset
    # chosen_dataset = Dataset.from_list(chosen_dataset)
    # map_kwargs["desc"] = f"Packing {dataset_name} dataset"
    # chosen_dataset = chosen_dataset.map(
    #         pack_examples, batched=True, fn_kwargs={"seq_length": 512}, **map_kwargs
    #     )
    print("finish packing chosen_dataset\n")
    # need to finish after vacation
    # #??????packing, ???????????????????
    # #?????{'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8]], 'attention_mask': [[0, 1, 1, 0, 0], [1, 1, 1]]}
    # #?????  chosen_dataset, ?list, ???????dict('input_ids':xx, 'attention_mask':xx)
    # chosen_dataset_final=[]
    # for i, tmp_data in enumerate(chosen_dataset):#??current_dataset????? the purpose of this action is fetch data from random selection order
    #     # chosen_dataset.append({'input_ids':torch.Tensor(tmp_data['input_ids']), 'attention_mask':torch.Tensor(tmp_data['attention_mask'])})
    #     # chosen_dataset.append(tmp_data['input_ids'])
    #     # if len(tmp_data['input_ids']) < 512:
    #     #     break
    #     chosen_dataset_final.append({'input_ids':torch.tensor(tmp_data['input_ids']), 'attention_mask':torch.tensor(tmp_data['attention_mask'])})
    # print("finish creating PromptDataset\n")
    # #here needs to consider when input_ids and labels are shorter than max_seq_len, but input_ids will be padded in dataloader, while label not
    # #this part needs to reconsideration
    # train_dataset = PromptDataset(prompt_dataset, chosen_dataset_final, reject_dataset,
    #                      tokenizer.pad_token_id, train_phase)
    # print("finished create_dataset_2() function")

    return train_dataset

def pack_examples(examples: dict[str, list[list]], seq_length: int) -> dict [str, list[list]]:
    """
    Pack examples into chunks of size `seq_length`.

    Args:
        examples (`dict[str, list[list]]`):
            Dictionary of examples with keys as strings and values as lists of lists.
        seq_length (`int`):
            Maximum sequence length.

    Returns:
        `dict[str, list[list]]`: Dictionary of examples with keys as strings and values as lists of lists.

    Example:

    ```python
    >>>
    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> pack_examples(examples, seq_length=5)
    {'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8]], 'attention_mask': [[0, 1, 1, 0, 0], [1, 1, 1]]}
    >>> pack_examples(examples, seq_length=2)
    {'input_ids': [[1, 2], [3, 4], [5, 6], [7, 8]], 'attention_mask': [[0, 1], [1, 0], [0, 1], [1, 1]]}
    ```
    """
    # Join  all the values into a single list
    examples = {k: sum(v, []) for k, v in examples.items()}
    # Split the values into chunks of size seq_length
    examples = {k: [v[i : i + seq_length] for i in range(0, len(v), seq_length)] for k, v in examples.items()}
    return examples



#????train_phase=1???
#?create_dataset_2()??????enumerate, ????, ????index
def create_dataset_3(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, rebuild):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    print("finished get raw_data\n")
    #train dataset
    train_dataset = raw_dataset.get_train_data()
    map_kwargs = {}
    map_kwargs["num_proc"] = 10 #here is the parallel process number
    map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
    train_dataset = train_dataset.map(maybe_apply_chat_template,fn_kwargs={"tokenizer": tokenizer, "is_output_dict":True},remove_columns="messages" if "messages" in train_dataset.column_names else None, **map_kwargs)

    print("finish maybe_apply_chat_template() to all data\n")
    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
    train_dataset = train_dataset.map(lambda ex: tokenizer(ex["text"]), **map_kwargs)

    print("finish tokenizering all data\n")


    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    # train_dataset = train_dataset.select_columns(['input_ids', 'attention_mask'])
    train_dataset = train_dataset.select_columns(['input_ids'])

    #no longer needed
    # train_dataset = train_dataset.map(lambda ex: {'chosen': {"input_ids":ex["input_ids"], "attention_mask":ex["attention_mask"]}})

    # print("finish capsulating all data\n")
    # from datasets.arrow_dataset import Dataset
    # chosen_dataset = Dataset.from_list(train_dataset['chosen'])
    # map_kwargs["desc"] = f"Applying packing to {dataset_name} dataset"
    train_dataset = train_dataset.map(
            pack_examples, batched=True, fn_kwargs={"seq_length": 512}, **map_kwargs
        )
    print("finish packing all data\n")
    chosen_dataset_final=[]
    for i, tmp_data in enumerate(chosen_dataset):#??current_dataset????? the purpose of this action is fetch data from random selection order
        chosen_dataset_final.append({'input_ids':torch.tensor(tmp_data['input_ids']), 'attention_mask':torch.tensor(tmp_data['attention_mask'])})
    print("finish capsulating all packed data\n")
    train_dataset = PromptDataset(prompt_dataset, chosen_dataset_final, reject_dataset,
                         tokenizer.pad_token_id, train_phase)
    print("finish creating PromptDataset\n")
    return train_dataset

def create_prompt_dataset_0(
        is_eval,
        local_rank,
        data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        end_of_conversation_token,
        sft_only_data_path):
    if is_eval:
        train_dataset, eval_dataset = create_prompt_dataset(
            local_rank,
            data_path,
            data_split,
            data_output_path,
            train_phase,
            seed,
            tokenizer,
            max_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
            sft_only_data_path=sft_only_data_path)
        return train_dataset, eval_dataset
    else:
        train_dataset = create_prompt_dataset_2(
            local_rank,
            data_path,
            data_split,
            data_output_path,
            train_phase,
            seed,
            tokenizer,
            max_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
            sft_only_data_path=sft_only_data_path)
        return train_dataset, None


def create_prompt_dataset_grpo(
        is_eval,
        local_rank,
        data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        end_of_conversation_token,
        sft_only_data_path):
    # if is_eval:
    #     train_dataset, eval_dataset = create_prompt_dataset(
    #         local_rank,
    #         data_path,
    #         data_split,
    #         data_output_path,
    #         train_phase,
    #         seed,
    #         tokenizer,
    #         max_seq_len,
    #         end_of_conversation_token=tokenizer.eos_token,
    #         sft_only_data_path=sft_only_data_path)
    #     return train_dataset, eval_dataset
    # else:
    train_dataset = create_prompt_dataset_2_grpo(
        local_rank,
        data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        sft_only_data_path=sft_only_data_path)
    return train_dataset, None

def create_prompt_dataset_2_grpo(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False):
    """
    Creates the prompt dataset
    """

    ################
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    #tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    #depends on tokenizer, as need tokenzier text
    fname = f"grpo_raw_only_train_full_{fname}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"

    cache_found = os.path.isfile(train_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)
    ####

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print(f'Creating grpo dataset {data_path}, {reload=}')

        if len(data_path) == 1:  # Single dataset.
            train_dataset = create_dataset_grpo(
                local_rank,
                data_path[0],
                data_split,
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload)
        print(f'finish read dataset')
        torch.save(train_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname, weights_only=False) # modifed 20250402 , weights_only=true


def create_dataset_grpo(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, rebuild):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    print("finished get raw_data\n")
    #train dataset
    train_dataset = raw_dataset.get_train_data()
#    train_dataset = train_dataset.select_columns(['messages'])
    del raw_dataset

    def make_conversation(example, prompt_column: str = 'problem'):
        prompt = []
        system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
        prompt.append({"role": "system", "content": system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}



    map_kwargs = {}
    map_kwargs["num_proc"] = 52 #here is the parallel process number
    map_kwargs["desc"] = f"Applying making conversation template to {dataset_name} dataset"
    train_dataset = train_dataset.map(make_conversation,**map_kwargs)

    print("finish making conversation template() to all data\n")

    # train_dataset = train_dataset.select_columns(['text'])
    # map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
    # map_kwargs["num_proc"] = 52
    # train_dataset = train_dataset.map(lambda ex: tokenizer(ex["text"]), **map_kwargs)
    # print("finish tokenizering all data\n")
    #
    # train_dataset = train_dataset.select_columns("input_ids")
    # map_kwargs["desc"] = f"Packing {dataset_name} dataset"
    # train_dataset = train_dataset.map(
    #     pack_examples, batched=True, fn_kwargs={"seq_length": max_seq_len}, **map_kwargs
    # )
    # print("finish packing chosen_dataset\n")
    return train_dataset
