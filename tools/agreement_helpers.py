import rbo 
import numpy as np 

# all=False - means we compute only for Top K sentences 
def global_rbo_agreement(all_model_ranks, k=3, p=0.7, all=True):
    num_models = len(all_model_ranks)
    num_queries = len(all_model_ranks[0])

    query_scores = []

    for q in range(num_queries):
        if all == True:
          model_tops = [all_model_ranks[m][q] for m in range(num_models)]
        else:
          model_tops = [all_model_ranks[m][q][:k] for m in range(num_models)]


        #Calculate RBO for every unique pair of models
        pair_scores = []
        for i in range(num_models):
            for j in range(i + 1, num_models):
                score = rbo.RankingSimilarity(np.argsort(model_tops[i]).tolist(), np.argsort(model_tops[j]).tolist()).rbo_ext(p=0.7)
                pair_scores.append(score)

        query_scores.append(np.mean(pair_scores))

    return np.mean(query_scores)

# M1/M2 are datasets objects & method refers to the two methods compare
def pair_agreement(M1, M2, method='M1 vs M2'):
  results = global_rbo_agreement([list(M1['ranks']), list(M2['ranks'])], all=True)
  print(f"RBO / {method}: {results:.4f}")

# M1/M2/M3 are datasets objects
def global_agreement(M1, M2, M3, all=True):
  results = global_rbo_agreement([list(M1['ranks']), list(M2['ranks']), list(M3['ranks'])], all=True)
  print(f"Global RBO: {results:.4f}")
