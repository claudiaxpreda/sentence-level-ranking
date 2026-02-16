import torch
import json
import numpy as np
from scipy.stats import rankdata


def get_indexes_context(context, skip, tokenizer):
  context_len = len(tokenizer(context)['input_ids'])
  start_indx = skip - 1
  end_indx = start_indx + context_len

  return start_indx, end_indx

def get_indexes_answer(answer, inputs, tokenizer):
  end_indx = len(inputs)
  answer_len = len(tokenizer(answer)['input_ids'])
  start_indx = end_indx - answer_len
  return start_indx, end_indx

def get_sents_indexes(sents, start_context, tokenizer):
  start_indx = start_context
  indexes = []
  for sent in sents:
    sent_len = len(tokenizer(sent)['input_ids'])
    end_indx = start_indx + sent_len
    indexes.append((start_indx, end_indx))
    start_indx = end_indx

  return indexes

def compute_attention(input_prompt_ids, model):
    all_input_ids = input_prompt_ids['input_ids']
    all_attention_maks = input_prompt_ids['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=all_input_ids, attention_mask=all_attention_maks,
                        output_attentions=True, return_dict_in_generate=True)

    output_attentions = outputs.attentions
    return output_attentions

def compute_attention_score_per_sentence(attention, input_prompt, contents, answers, sentences, skip_entry, tokenizer):
  batch_scores = []
  for indx, ctx in enumerate(contents):
    skip_padding = torch.count_nonzero(input_prompt['input_ids'][indx] == tokenizer.pad_token_id).item()
    scores = attention[:, indx]
    scores = scores[:, :,skip_padding:, skip_padding:]
    #print(scores.shape)
    scores = scores.mean(dim=0).mean(dim=0)
    #print(scores.shape)
    #print(scores)

    inputs = input_prompt['input_ids'][indx][skip_padding:]
    #print(inputs)
    ctx_start, ctx_end = get_indexes_context(ctx, skip_entry, tokenizer)
    ans_start, ans_end = get_indexes_answer(answers[indx], inputs, tokenizer)
  # print(ans_start, ans_end)
  # print(tokenizer.decode(inputs[ans_start:ans_end]))

    sents_indx = get_sents_indexes(sentences[indx], ctx_start, tokenizer)

    sents_score_0 = [scores[ans_start : ans_end, sent_start : sent_end].sum(dim=1) for
                 (sent_start, sent_end) in sents_indx]


    sents_score = [torch.max(score).item() for score in sents_score_0]

    # sents_score_0 = [torch.max(scores[ans_start : ans_end, sent_start : sent_end], dim=1).values for
    #              (sent_start, sent_end) in sents_indx]


    # sents_score = [torch.sum(score, dim=0).item() for score in sents_score_0]


    batch_scores.append(sents_score)

  return batch_scores


def process_attention_scores_output(all_scores):
    att_scores = []
    positions = []
    ranks = []

    for val in all_scores:
        keys = [str(x) for x in range(len(val))]
        entry = dict(zip(keys, val))
        position = np.argmax(val).item() + 1
        att_scores.append(json.dumps(entry))

        rank = rankdata(val, method='ordinal')
        rank = len(rank) + 1 - rank
        positions.append(position)
        ranks.append(rank.tolist())
    
    return att_scores, positions, ranks