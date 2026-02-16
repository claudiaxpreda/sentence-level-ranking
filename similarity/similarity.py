import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from dotenv import dotenv_values
import pandas as pd
from similarity_helpers import *
from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SPLIT = sys.argv[1]
DATASET_NAME = sys.argv[2]
MODEL_NAME_CONVERT = sys.argv[3]
BATCH_SIZE = sys.argv[4]
PATH_BASE = sys.argv[5]
METHOD = 'M1'
MODEL_NAME_BLEURT = 'lucadiliello/BLEURT-20'

tokens = dotenv_values(".env") 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_convert = AutoTokenizer.from_pretrained(MODEL_NAME_CONVERT, padding_side='left')
tokenizer_convert.add_special_tokens({"pad_token": "151647"})
model_convert = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_CONVERT,
    dtype="auto",
    device_map="auto",
)

model_bluert = BleurtForSequenceClassification.from_pretrained(MODEL_NAME_BLEURT).to(device)
tokenizer_bleurt = BleurtTokenizer.from_pretrained(MODEL_NAME_BLEURT)

start_time = start_session_details(SPLIT, DATASET_NAME, BATCH_SIZE, METHOD)

dataset, columns = get_dataset(dataset_name, split)
dataset, columns = split_context_sentences(dataset, columns)

dataset = dataset.map(
    lambda e: {
        'chats': 
            chat_template(e[columns[1]], e[columns[2]])
        })

dataset_len = len(dataset)
chats = list(dataset['chats'])
sentences= list(dataset['sentences'])

statements = []
bluert_scores = []

for i in tqdm(range(0, len(chats), BATCH_SIZE)):
    end_interval = min(i+BATCH_SIZE, len(chats))
    chunk_sents = sentences[i:end_interval]
    chat_inputs = tokenizer_convert.apply_chat_template(
                        chats[i:end_interval],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
    
    model_inputs = tokenizer_convert(chat_inputs, padding='longest', return_tensors="pt").to(model_convert.device)
    generated_ids = model_convert.generate(**model_inputs, max_new_tokens=1000)

    contents = [
        tokenizer_convert.decode(
            generated_ids[indx][len(model_inputs.input_ids[indx]):],
            skip_special_tokens=True) for indx in range(generated_ids.size()[0])
    ]

    targets = [ [contents[indx]] * len(chunk_sents[indx]) for indx in range(len(chunk_sents))]
    targets = sum(targets, [])
    references = sum(chunk_sents, [])

    with torch.no_grad():
        inputs = tokenizer_bleurt(
            references, targets, padding='longest', return_tensors='pt').to(device)
        res = model_bluert(**inputs).logits.flatten().tolist()
    
    statements += contents
    bluert_scores += res

dataset = dataset.add_column('statement', statements)
columns.append('statement')

scores, positions, ranks = process_scores_output(bluert_scores, sentences)

dataset = dataset.add_column('score', scores)
columns.append('score')

dataset = dataset.add_column('position', positions)
columns.append('position')

dataset = dataset.add_column('ranks', positions)
columns.append('position')

dataset = dataset.select_columns(columns)

end_session_details(start_time, dataset_len)

dataframe = dataset.with_format('pandas')
dataframe.to_csv(PATH_BASE.format(split=split), index=False)

dataframe.to_csv(PATH_BASE + f'{SPLIT}_{METHOD}_{DATASET_NAME}.csv', index=False)
