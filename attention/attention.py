import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

import torch
import pandas as pd
import attention_helpers as helpers

from dotenv import dotenv_values
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from prompts import PROMPT_QA, PROMPT_START

SPLIT = sys.argv[1]
DATASET_NAME = sys.argv[2]
MODEL_NAME = sys.argv[3]
BATCH_SIZE = sys.argv[4]
PATH_BASE = sys.argv[5]
METHOD = 'M3'

tokens = dotenv_values(".env") 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token = tokens['HUGGING_FACE_TOKEN'], padding_side="left")
tokenizer.add_special_tokens({"pad_token": "151647"})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    attn_implementation="eager",
    token = tokens['HUGGING_FACE_TOKEN'],
    torch_dtype=torch.bfloat16,
)

start_time = start_session_details(split, DATASET_NAME, BATCH_SIZE, METHOD)

dataset, columns = get_dataset(DATASET_NAME, split)
dataset, columns = split_context_sentences(dataset, columns)

dataset= dataset.map(lambda e: {'prompts': 
                                PROMPT_QA.format(context=e[columns[0]],
                                                    question=e[columns[1]],
                                                        answer=e[columns[2]])})

dataset = dataset.map(lambda e : {columns[0] : " ".join(e['sentences'])})

dataset_len = len(dataset)

skip_entry = len(tokenizer.tokenize(PROMPT_START))
contents = list(dataset[columns[0]])
answers = list(dataset[columns[2]])
sentences = list(dataset['sentences'])
prompts=list(dataset['prompts'])

all_scores = []

for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    end_interval = min(i+BATCH_SIZE, len(prompts))
    input_prompt = prompts[i:end_interval]
    input_prompt_ids = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(model.device)

    attention = helpers.compute_attention(input_prompt_ids, model)
    attention = torch.stack(attention)[17:37]
    batch_contents = contents[i:end_interval]
    batch_answers = answers[i:end_interval]
    batch_sentences = sentences[i:end_interval]
    batch_scores = helpers.compute_attention_score_per_sentence(attention,input_prompt_ids, batch_contents, batch_answers, batch_sentences, skip_entry, tokenizer)
    all_scores += batch_scores

att_scores, positions, ranks = helpers.process_attention_scores_output(all_scores)

dataset= dataset.add_column('attention_score', att_scores)
columns.append('attention_score')

dataset = dataset.add_column('position', positions)
columns.append('position')

dataset = dataset.add_column('ranks', ranks)
columns.append('ranks')

dataset =  dataset.select_columns(columns)

end_session_details(start_time, dataset_len)


dataframe = dataset.with_format('pandas')
dataframe.to_csv(PATH_BASE + f'{SPLIT}_{METHOD}_{DATASET_NAME}.csv', index=False)

