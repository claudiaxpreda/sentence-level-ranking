import sys
import os
import json as json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))
from prompts import get_qa_to_statement_user_prompt, get_qa_to_statement_system_prompt
from scipy.stats import rankdata


def chat_template(question, answer): 
    chat = [
        {'role': 'system', 'content': get_qa_to_statement_system_prompt() },
        {'role': 'user', 'content': get_qa_to_statement_user_prompt(question, answer)}
    ]
    return chat

def process_scores_output(bleurt_scores, sentences):
    scores = []
    positions = []
    ranks= []
    cursor = 0

    for sents in sentences:
        val = len(sents)
        score_sents = bleurt_scores[cursor : (cursor + val)]
        cursor = cursor + val
        keys = [str(x + 1) for x in range(val)]
        entry = dict(zip(keys, score_sents))
        max_elem = max(score_sents)
        max_elem_indx = score_sents.index(max_elem)
        scores.append(json.dumps(entry))
        positions.append(max_elem_indx + 1)
        rank = rankdata(score_sents, method='ordinal')
        rank = len(rank) + 1 - rank
        ranks.append(rank)

    return (scores, positions, ranks)