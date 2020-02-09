from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from malss import MALSS


for name, data in [('iris', load_iris()), 
                    ('wine', load_wine()),
                    ('breast', load_breast_cancer())]:
    model = MALSS(task='clustering', lang='en')
    model.fit(data.data, None, f'report_{name}')

# dataset       number_of_clusters ground truth
# iris          3                  3
# wine          3                  3
# breast_cancer 2                  2