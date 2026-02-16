import sys
import os
import torch


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

import pandas as pd
from dotenv import dotenv_values
from utils import *
from sentences_helpers import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


SPLIT = sys.argv[1]
DATASET_NAME = sys.argv[2]
MODEL_NAME = sys.argv[3]
BATCH_SIZE = sys.argv[4]
PATH_BASE = sys.argv[5]
METHOD = 'M2'


tokens = dotenv_values(".env") 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token = tokens['HUGGING_FACE_TOKEN'], padding_side="left")
tokenizer.add_special_tokens({"pad_token": "151647"})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto", 
    token = tokens['HUGGING_FACE_TOKEN'],
    torch_dtype=torch.bfloat16,

)

start_time = start_session_details(SPLIT, DATASET_NAME, BATCH_SIZE, METHOD)

dataset, columns = get_dataset(dataset_name, split)
dataset, columns = split_context_sentences(dataset, columns)
dataset = create_prompts_dataset(dataset, columns)
dataset_len = len(dataset)

no_sents = list(dataset['no_sents'])
prompts = list(dataset['prompts'])
prompts_flatten = sum(prompts, [])
prompts_labels = list(dataset['prompts_labels'])
prompts_flatten_label = sum(prompts_labels, [])

losses = []

for i in tqdm(range(0, len(prompts_flatten), BATCH_SIZE)):
    end_interval = min(i+BATCH_SIZE, len(prompts_flatten))
    inputs_label = prompts_flatten_label[i:end_interval]
    inputs = prompts_flatten[i:end_interval]
    loss_item = qloss_func(inputs_label, inputs, tokenizer, model, device, loss_fn)
    losses += loss_item


loss_scores = process_loss_scores_output(losses, no_sents)
dataset = dataset.add_column('loss_score', loss_scores)
columns.append('loss_score')

scores, positions, ranks = process_scores_output(losses, no_sents)
dataset = dataset.add_column('score', scores)
columns.append('score')

dataset = dataset.add_column('position', positions)
columns.append('position')

dataset = dataset.add_column('ranks', ranks)
columns.append('ranks')

end_session_details(start_time, dataset_len)

dataset = dataset.select_columns(columns)

dataframe = dataset.with_format('pandas')
dataframe.to_csv(PATH_BASE + f'{SPLIT}_{METHOD}_{DATASET_NAME}.csv', index=False)