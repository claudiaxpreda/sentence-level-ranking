import spacy
import re
import time


from datasets import load_dataset
from dotenv import dotenv_values


tokens = dotenv_values(".env") 


SQUAD2_DT = 'SQUAD2'
FTQA_DT = 'FTQA'

DATASET_MAP = {
    'FTQA' : 'GEM/FairytaleQA', 
    'SQUAD2' : 'rajpurkar/squad_v2'
}

DATASET_COLUMNS = {
    'FTQA' : ['content', 'question', 'answer', 'local_or_sum', 'ex_or_im'],
    'SQUAD2' : ['context', 'question', 'answers']
}

nlp = spacy.load("en_core_web_lg")

def get_dataset(name, split): 

    if name == FTQA_DT:
        dataset = load_dataset(DATASET_MAP[name], split=split, trust_remote_code=True, token=tokens['HUGGING_FACE_TOKEN'])
        dataset = dataset.map(lambda e: {'question': e['question'].capitalize()})
        dataset = dataset.map(lambda e: {'answer': e['answer'].capitalize()})

        dataset = dataset.map(
            lambda e: {'content': re.sub(r'\s+([?.!,;"])', r'\1', e['content'])})
        dataset = dataset.map(
            lambda e: {'question': re.sub(r'\s+([?.!,;"])', r'\1', e['question'])})
        dataset = dataset.map(
            lambda e: {'answer': re.sub(r'\s+([?.!,;"])', r'\1', e['answer'])})
    else: 
        dataset = load_dataset(DATASET_MAP[name], split=split)

    dataset = dataset.select_columns(DATASET_COLUMNS[name])
    columns = DATASET_COLUMNS[name]

    if name == SQUAD2_DT:
        dataset = dataset.filter(lambda e: e['answers']['text'] != [])
        dataset = dataset.map(lambda e: {'answers_start': e['answers']['answer_start'][0]})
        dataset = dataset.map(lambda e: {'answers': e['answers']['text'][0]})
        columns.append('answers_start')
        


    return (dataset, columns)

def split_paragraph(text): 
    doc = nlp(text)
    sentences = [sent.text.capitalize() for sent in doc.sents]
    return sentences

def split_context_sentences(dataset, columns):
    dataset = dataset.map(
            lambda e: {'sentences': split_paragraph(e[columns[0]])})
    dataset = dataset.map( 
            lambda e: {'no_sents': len(e['sentences'])})
    
    columns.append('no_sents')
    columns.append('sentences')
    
    return dataset, columns


def start_session_details(split, dataset_name, batch_size , method): 
    start_time = time.localtime()
    print('Program start time: {}\n'.format(time.strftime("%H:%M:%S", start_time)))
    print ('Method: {} // Split: {} // dataset: {} // batch: {} \n'.format(method, split, dataset_name, batch_size))

    return start_time

def end_session_details(start_time, dataset_len):
    end_time = time.localtime()
    duration = time.mktime(end_time) - time.mktime(start_time)
    
    print('Program end time: {}\n'.format(time.strftime("%H:%M:%S", end_time)))
    print('Program running time : {} \\ Number of questions {} \n'.format(duration, dataset_len))
    print ('Average time: {}\n'.format(duration / dataset_len))
