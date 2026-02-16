import torch
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

import numpy as np
from prompts import generate_prompt_qa, generate_prompt_qa_labels
from scipy import special
from scipy.stats import rankdata



def qloss_func(prompts_with_label, prompts, tokenizer, model, device, loss_fn):

  input_prompt = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
  whole_prompt = tokenizer(prompts_with_label, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
   
  #Add the eos token to the whole prompts, for the whole batch
  whole_prompt['input_ids'] = torch.cat((whole_prompt['input_ids'], torch.tensor([[tokenizer.eos_token_id]] * whole_prompt['input_ids'].shape[0], device=device)), dim=1)
  whole_prompt['attention_mask'] = torch.cat((whole_prompt['attention_mask'], torch.tensor([[1]] * whole_prompt['attention_mask'].shape[0], device=device)), dim=1)
  
  with torch.no_grad():
    outputs = model(input_ids=whole_prompt['input_ids'], attention_mask=whole_prompt['attention_mask'])
    logits = outputs.logits

  batch_losses = []
  for logit, input, whole in zip(logits, input_prompt['input_ids'], whole_prompt['input_ids']):
    # Remove padding
    padding = torch.count_nonzero(whole == tokenizer.pad_token_id)
    whole = whole[padding:]
    padding = torch.count_nonzero(input == tokenizer.pad_token_id)
    input = input[padding:]

    # Remove the last logit (unnecessary, automatically added by the model)
    logit = logit[:-1]

    # Get from the logits just the ones corresponding to the actual generation (label)
    good_logit = logit[-(len(whole) - len(input)):]

    # Get the label
    good_label = whole[len(input):]


    loss = loss_fn(
        good_logit,
        good_label,
    )

    batch_losses.append(loss.item())

  return batch_losses

def create_prompts_dataset(dataset, columns): 
  dataset = dataset.map(
            lambda e: {
                'prompts': generate_prompt_qa(
                    e[columns[0]], e['sentences'], e[columns[1]])})
  dataset = dataset.map(
            lambda e: {
                'prompts_labels': generate_prompt_qa_labels(
                                    e[columns[0]], e['sentences'],
                                        e[columns[1]], e[columns[2]])})
  return dataset

def process_loss_scores_output(losses, no_sents):
  scores = []
  cursor = 0
  for val in no_sents:
    loss_vals = losses[cursor: (cursor + val + 1)]
    cursor = cursor + val + 1
    keys = [str(x) for x in range(val + 1)]
    entry = dict(zip(keys, loss_vals))
    scores.append(json.dumps(entry))
  
  return scores


def process_scores_output(losses, no_sents):
  scores = []
  positons = []
  ranks = []
  cursor = 0
  
  for val in no_sents:
    context_log_prob = losses[cursor]
    log_probs = np.array(losses[(cursor + 1) : (cursor + val + 1)])
    log_probs_diff = log_probs - context_log_prob
    log_scores = special.softmax(log_probs_diff)
    max_value = log_scores.max()
    position = np.where(log_scores == max_value)[0] + 1
    keys = [str(x) for x in range(val + 1)]
    entry = dict(zip(keys, log_scores.tolist()))
    rank = rankdata(log_scores.tolist(), method='ordinal')
    rank = len(rank) + 1 - rank
    scores.append(json.dumps(entry))
    positons.append(position[0])
    ranks.append(rank)
    cursor = cursor + val + 1

  return  scores, positons, ranks