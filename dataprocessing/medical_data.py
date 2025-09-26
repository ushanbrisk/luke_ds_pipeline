import json
from datasets import load_dataset




def process_func2(example, prompts, tokenizer):
    PROMPT=prompts
    instruction = example['instruction']
    input1 = example['input']
    output = example['output']
    messages = [
        {'role': 'system', 'content': f"{PROMPT}"},
        {'role': 'user', 'content': input1},
        {'role': 'assistant', 'content': output},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        truncate=True,
        #     return_tensors='pt',
        #     enable_thinking=False

    )
    #     example['text'] = text
    return {"text" :text}

def process_func(example, tokenizer, max_length):
    MAX_LENGTH=max_length
    PROMPT="你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"

    input = example["question"]
    think = example["think"]
    answer = example["answer"]
    output = f"<think>{think}</think> \n {answer}"

    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{output}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # ?????
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def prepare_medical_dataset(dataset_name,tokenizer,max_length):

    dataset = load_dataset(dataset_name)

    train_dataset = dataset['train']
    eval_dataset = dataset['eval']

    map_kwargs = {}
    map_kwargs["num_proc"] = 52  # here is the parallel process number
    map_kwargs["desc"] = f"Applying chat template to train dataset"
    train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names,fn_kwargs={"tokenizer": tokenizer, "max_length":max_length}, **map_kwargs)
    map_kwargs = {}
    map_kwargs["num_proc"] = 52  # here is the parallel process number
    map_kwargs["desc"] = f"Applying chat template to eval dataset"
    eval_dataset  = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names,fn_kwargs={"tokenizer": tokenizer, "max_length":max_length}, **map_kwargs)

    return train_dataset, eval_dataset
