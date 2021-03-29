import pandas as pd
import numpy as np

df = pd.read_csv("Train.csv")
#print(df.head())
airport = df[df['Category'] == 0].values
#print(len(np.unique(arr)))
print(len(airport))

from collections import defaultdict

class_freq = defaultdict(int)
freq = []
for ind, val in df.iterrows():
    class_freq[int(val["Category"])] += 1


for ind, val in class_freq.items():
    freq.append([ind, val])
#print(freq)
freq.sort()
#print(freq)
#exit()
freq = [i[1] for i in freq]

wt_per_class = [0.] * 67
N = float(sum(freq))
print(freq)
print(N)

for i in range(67):
    wt_per_class[i] = N / float(freq[i])

print(wt_per_class)
exit()
weight = [0] * len(df)
for ind, val in df.iterrows():
    cat = val["Category"]
    weight[ind] = wt_per_class[cat]
