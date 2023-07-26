import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def compute_tsne_embeddings(data: dict, perplexity: int = 50, n_iter: int = 3000, average_over: str = 'repetitions') -> np.ndarray:
    """
    Compute t-sne embeddings of the data
    """

    if average_over == 'videos':
        dim = 0
    elif average_over == 'repetitions':
        dim = 1
    elif average_over == 'voxels':
        dim = 2
    else:
        raise ValueError('average_over must be one of "videos", "repetitions" or "voxels"')

    avg_fmri = np.mean(data['train_data'], axis=dim)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(avg_fmri)

    return tsne_results

def t_sne_visualization(data: dict, perplexity: int = 50, n_iter: int = 3000, average_over: str = 'repetitions') -> None:
    """
    Plot t-sne visualization of the data
    """

    tsne_results = compute_tsne_embeddings(data, perplexity=perplexity, n_iter=n_iter, average_over=average_over)

    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('t-sne dim 1')
    plt.ylabel('t-sne dim 2')
    plt.show()
