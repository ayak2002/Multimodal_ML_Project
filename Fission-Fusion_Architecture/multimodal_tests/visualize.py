import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load saved feature vectors
shared = np.load('/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/multimodal_tests/snapshots/feat_outputs/2025-Apr-29_seed436196_sccid/features/Allen/shared_features.npy')
specific = np.load('/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/multimodal_tests/snapshots/feat_outputs/2025-Apr-29_seed436196_sccid/features/Allen/specific_features.npy')
fused = np.load('/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/multimodal_tests/snapshots/feat_outputs/2025-Apr-29_seed63940_sccid/features/Allen/features.npy')
print("Shared shape:", shared.shape)
print("Specific shape:", specific.shape)
print("Mean of shared embeddings:", np.mean(shared))
print("Mean of specific embeddings:", np.mean(specific))
print("Are shared and specific identical?", np.allclose(shared, specific))
print("Loaded shared shape:", shared.shape)
print("Loaded specific shape:", specific.shape)

# Example 1: visualize shared vs specific with TSNE

def plot_tsne(shared, specific):
  
    N = 2000
    idx = np.random.choice(len(shared), N, replace=False)
    X = np.vstack([shared[idx], specific[idx]])
    y = np.array([0]*N + [1]*N)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(X_embedded[y==0, 0], X_embedded[y==0, 1], label='Shared', alpha=0.5)
    plt.scatter(X_embedded[y==1, 0], X_embedded[y==1, 1], label='Specific', alpha=0.5)
    plt.legend()
    plt.title("t-SNE of Shared vs Specific Features")
    plt.show()



# Example 2: cosine similarity between shared and specific
def plot_cosine_similarity(shared, specific):
    sims = []
    for i in range(len(shared)):
        sim = cosine_similarity(shared[i].reshape(1, -1), specific[i].reshape(1, -1))[0][0]
        sims.append(sim)

    plt.figure(figsize=(8, 6))
    plt.hist(sims, bins=50)
    plt.title("Cosine Similarity between Shared and Specific Embeddings")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.show()

# Call functions
plot_tsne(shared, specific)
plot_cosine_similarity(shared, specific) 