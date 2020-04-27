from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from malss import MALSS


def test_malss():
    for name, data in [('iris', load_iris()), 
                        ('wine', load_wine()),
                        ('breast', load_breast_cancer())]:
        model = MALSS(task='clustering', lang='en')
        model.fit(data.data, None, f'report_{name}')

    # dataset       number_of_clusters ground truth
    # iris          3                  3
    # wine          3                  3
    # breast_cancer 2                  2

def test_malss2():
    import numpy as np
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    for name, data, nc, pattern in [('iris', load_iris(), 3, [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]), 
                                    ('wine', load_wine(), 3, [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]),
                                    ('breast', load_breast_cancer(), 2, [[0, 1], [1, 0]])]:
        print(name)
        model = MALSS(task='clustering', lang='en', verbose=False, shuffle=False)
        model.fit(data.data)
        print('k-means')
        est = model.algorithms[0].estimator
        est.n_clusters = nc
        pred = est.fit_predict(model.data.X)
        max_acc = 0
        for ptn in pattern:
            acc = np.mean(np.array(data.target) == np.array([ptn[i] for i in pred]))
            if acc > max_acc:
                max_acc = acc
        print(max_acc)
        print('Hierarchical clustering')
        pred = fcluster(model.algorithms[1].estimator, t=nc, criterion='maxclust') - 1
        max_acc = 0
        for ptn in pattern:
            acc = np.mean(np.array(data.target) == np.array([ptn[i] for i in pred]))
            if acc > max_acc:
                max_acc = acc
        print(max_acc)

    # dataset       k-means Hierarchical clustering 
    # iris          0.833   0.787
    # wine          0.972   0.837
    # breast_cancer 0.905   0.631


if __name__ == '__main__':
    test_malss()