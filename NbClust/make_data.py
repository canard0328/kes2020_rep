from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd

for name, data in [('iris', load_iris()), ('wine', load_wine()), ('cancer', load_breast_cancer())]:
    data_pd = pd.DataFrame(data.data, columns=data.feature_names)
    data_pd['target'] = data.target_names[data.target]
    data_pd.to_csv(f'{name}.csv', index=False)