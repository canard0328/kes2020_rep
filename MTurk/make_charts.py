import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def cilen(arr, alpha=0.95):
    if len(arr) <= 1:
        return 0
    m, e, df = np.mean(arr), stats.sem(arr), len(arr) - 1
    interval = stats.t.interval(alpha, df, loc=m, scale=e)
    cilen = np.max(interval) - np.mean(interval)
    return cilen

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def bar_plot():
    df = pd.read_csv('result.csv')

    all_before = df['before'].mean()
    over_before = df[df['experience']=='over1']['before'].mean()
    less_before = df[df['experience']!='over1']['before'].mean()
    b1 = [all_before, over_before, less_before]

    all_before_err = cilen(df['before'])
    over_before_err = cilen(df[df['experience']=='over1']['before'])
    less_before_err = cilen(df[df['experience']!='over1']['before'])
    e1 = [all_before_err, over_before_err, less_before_err]

    all_after = df['after'].mean()
    over_after = df[df['experience']=='over1']['after'].mean()
    less_after = df[df['experience']!='over1']['after'].mean()
    b2 = [all_after, over_after, less_after]

    all_after_err = cilen(df['after'])
    over_after_err = cilen(df[df['experience']=='over1']['after'])
    less_after_err = cilen(df[df['experience']!='over1']['after'])
    e2 = [all_after_err, over_after_err, less_after_err]

    bar_width = 0.25

    r1 = np.arange(len(b1))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, b1, color='dimgray', width=bar_width, label='Before', yerr=e1)
    plt.bar(r2, b2, color='lightgray', width=bar_width, label='After', yerr=e2)
    plt.xlabel('Examinees')
    plt.ylabel('Score')
    plt.xticks([r + bar_width / 2 for r in range(len(b1))], ['All', 'Over 1 year', 'Less than 1 year'])
    plt.legend(loc='upper left')
    plt.savefig('bar_before_after.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    bar_plot()