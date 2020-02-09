from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator
h2o.init()
print(h2o.__version__)  # 3.28.0.2


for data in [load_iris(), load_wine(), load_breast_cancer()]:
    df = pd.DataFrame(data.data)
    df.columns = data.feature_names
    h2o_data = h2o.H2OFrame(df)
    train, valid = h2o_data.split_frame(ratios=[.8], seed=0)
    kmeans = H2OKMeansEstimator(k=10, estimate_k=True, seed=0)
    kmeans.train(training_frame=train, validation_frame=valid)
    print(kmeans.summary())

# dataset       number_of_clusters ground truth
# iris          2                  3
# wine          3                  3
# breast_cancer 4                  2