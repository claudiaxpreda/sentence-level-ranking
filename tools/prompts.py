PROMPT_QA = 'Answer the following question based on the context.\nContext: {context}\nQuestion: {question}\n### Response: '
PROMPT_QA_LABELS = 'Answer the following question based on the context.\nContext: {context}\nQuestion: {question}\n### Response: {answer}'
PROMPT_START = 'Answer the following question based on the context.\nContext: '

PROMPT_QA_TO_STATEMENT_SYSTEM = 'You are a helpful assistant that rewrites question and answer pairs as factual statements using only the words from the original questions.'
PROMPT_QA_TO_STATEMENT_USER = '''
    You are given a question and answer pair.

    Your task is to convert the question and the answer in a single factual statement using only the words from the question and the answer. 

    STRICT RULES: 
    1. Use only the words present in the question and the answer.
    2. Do NOT add any new words or names. 
    3. Do NOT add any additional explanation of any kind. 
    4. Do NOT explain, justify, or rephrase any existing words.
    5. For YES/NO and TRUE/FALSE question convert the question into a statement without including the answer.

    Examples: 
    Q: When does the prince starts his journey? 
    A: Tomorrow 
    S: The prince starts his journey tomorrow. 

    Q: Is it going to rain today? 
    A: Yes 
    S: Today it is going to rain. 

    Now convert: 
    Q: {question}
    A: {answer}
    S:

'''

def process_sentences(sentences): 
    removed_sents = [ 
        sentences[:indx] + sentences [(indx + 1) :] 
            for indx in [*range(len(sentences))]
    ]

    texts = [" ".join(sentences)] + [" ".join(sents) 
                                        for sents in removed_sents]

    return texts

def generate_prompt_qa(context, sentences, question):
    texts = process_sentences(sentences)

    return [PROMPT_QA.format(
                context=text, question=question) for text in texts]

def generate_prompt_qa_labels(context, sentences, question, answer): 
    texts = process_sentences(sentences)

    return [
        PROMPT_QA_LABELS.format(
                context=text, question=question, answer=answer) for text in texts
    ]

def get_qa_to_statement_system_prompt():
    return PROMPT_QA_TO_STATEMENT_SYSTEM

def get_qa_to_statement_user_prompt(question, answer): 
    return PROMPT_QA_TO_STATEMENT_USER.format(question=question, answer=answer)