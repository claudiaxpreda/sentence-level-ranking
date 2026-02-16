# Sentence-Level Ranking: Visualizing Source Text Coverage Relative to Question-Answer Pairs

### Description
This repository propose three methods for determining the importance of a sentence in answerig a quenstion based on a given context: 

- Similarity Score: computes the similarity between each sentence of the context and the question
answer statement;

- Loss Probability Score: assigns a score based on the cross-entropy loss produced by removing the question from the source text and asking the model to answer the question. The score is computed using the following formula where S = Sentence, Q = Question, A = Answer, C = contex, and C-S = context without the evaluated sentence: $$ P(S,Q,A) = softmax(Loss(C-S, Q, A) - Loss(C, Q, A))$$

- Attention Score: calculates the attention scores for each sentence based on the most important tokens used to determine the answer.

### Project Structure

- Each method is found in their respecie file. 
- Common modules are found in the tools file. 
- In requirements.txt you will find the packages needed to run the solution. 
- Currently we support two datasets SQUAD2 and FairytaleQA. If you wish to add more datasets please see files tools/utils.py and follow the pattern. 
- The .env file is needed to interact with HuggingFace Hub (for example, for model's access). An example is provided in this project. 
- By default the new datasets are saved as cvs files based on the path provided. 
- The code has been tested only on Qwen3 family of models.

> [!WARNING]
> FairytaleQA needs to be read using an older version of the dataset (3.6.0) due to not being updated to the new format required. 

- See below how to run the files: 

```python

''' SPLIT: test, train, validation 
    DATASET_NAME: FTQA (FairytaleQA) and SQUAD2
    MODEL_NAME: Hugging Face model 
    BATCH_SIZE: 8 for attention, 32 for the others
    PATH_BASE: folder in which the output is saved
    METHOD: the name of the method, included in the 
            output file name; M1=Similarity, M2=Loss,
            M3=Attention.
'''

python3 method_dir/method.py SPLIT DATASET_NAME MODEL_NAME BATCH_SIZE PATH_BASE METHOD

# Example 

python3 attention/attention.py 'test' 'FTQA' 'Qwen/Qwen3-4B-Instruct-2507' '8' 'data_output/' 'M3'


