import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rbo

def get_quarter(val):
    if val <= 0.25: return "Q1"
    if val <= 0.50: return "Q2"
    if val <= 0.75: return "Q3"
    
    return "Q4"

def get_violin_quarter_distribution(dataset):
    ranks = []

    for indx, val in enumerate(list(dataset['ranks'])):
        args = np.argsort(val) + 1
        args = args.tolist()
        no_sents = dataset[indx]['no_sents']
        ranks.append(args[0]/no_sents)
        if len(args) >= 2:
            ranks.append(args[1]/no_sents)
        if len(args) >= 3:
            ranks.append(args[2]/no_sents)

    quarters = [get_quarter(p) for p in ranks]

    df = pd.DataFrame({
        'Position_Pct': ranks,
        'Quarter': quarters
    })

    # --- Plotting the Violin with Separator Lines ---

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.violinplot(
        data=df,
        x='Quarter',
        y='Position_Pct',
        order=["Q1", "Q2", "Q3", "Q4"],
        palette="viridis",
        inner="stick", # Shows the individual data points as lines
        saturation=0.8
    )

    # Add the vertical lines to separate the quarters
   
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)
    ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1.5)
    ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5)

    # Formatting
    plt.title("Distribution Top 3 Sentences on Local-Explicit QA Pairs", fontsize=20)
    plt.ylabel("Relative Position", fontsize=18)
    plt.xlabel("Source Text", fontsize=18)
    plt.ylim(0, 1) # Ensure the plot covers the full text range

    plt.show()


# M1, M2, M3 are datasets objects
def plot_correlation_matrix(M1, M2, M3):

    # Data Structure: Each method has a list of rankings (one per test case)
    method_data = {
        'Similarity': list(M1['ranks']),
        'Loss Probability': list(M2['ranks']),
        'Attention': list(M3['ranks'])
    }

    methods = list(method_data.keys())
    n_entries = len(method_data['Attention']) # Number of test cases
    p_value = 0.7

    # Initialize Matrix
    matrix_data = np.zeros((len(methods), len(methods)))

    # Calculate Mean RBO for each pair
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                matrix_data[i, j] = 1.0
            elif i < j: # Only calculate upper triangle for efficiency
                scores = []
                for k in range(n_entries):
                    list1 = np.argsort(method_data[m1][k]).tolist()
                    list2 = np.argsort(method_data[m2][k]).tolist()

                    # Calculate RBO Extrapolated for this specific entry
                    score = rbo.RankingSimilarity(list1, list2).rbo_ext(p=0.7)
                    scores.append(score)

                # Mean score across all entries
                avg_score = np.mean(scores)
                matrix_data[i, j] = avg_score
                matrix_data[j, i] = avg_score # Mirror it

    # 4. Visualize
    rbo_df = pd.DataFrame(matrix_data, index=methods, columns=methods)
    plt.figure(figsize=(8, 6))
    sns.heatmap(rbo_df, annot=True, cmap="Blues", vmin=0, vmax=1,)
    plt.title(f"Mean Pairwise RBO_EXT (p={p_value})", fontsize=16)
    plt.show()
