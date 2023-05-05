import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from params import *

print("loading file for explanation3")
start = time.time()
df = pd.read_csv(f"/home/miana/Downloads/weather_berlin/charts.csv")
total = df.groupby(by=["song", "artist"])

print("finished loading file in", time.time()-start, "seconds")

a = -np.log(ratio)
b = -a*np.exp(100*a) / (1 - np.exp(99*a))
T = int(len(total)*percentage_to_analyse)
print("starting explanation3")
start = time.time()
songs = []
n = 1
for name, group in total:
    if len(group.index) < 5:
        continue
    group["date"] = pd.to_datetime(group["date"])
    group = group.sort_values("date")
    group["aggr"] = 0
    S = 0
    for i in range(len(group.index)):
        group.iloc[i, group.columns.get_loc("aggr")] = S
        S += (b*np.exp(-a*group.iloc[i]["rank"])*100)
    songs.append(group)
    if n >= 25:
        print("reached", 25, "songs")
        break
    n += 1




fig, ax = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        k = sns.lineplot(x='aggr', y='rank', data=songs[i*5+j],  alpha  = 1, ax=ax[i,j])
        if i==4:
            k.set(xlabel='Exposure \n %week')
        else:
            k.set(xlabel='')
        if j==0:
            k.set(ylabel='Rank(T+1)')
        else:
            k.set(ylabel='')
        ax[i-1,j].invert_yaxis()


fig.suptitle("25 songs chosen at random")
fig.align_ylabels(ax[:, :])
fig.tight_layout()
print("finished explanation3 in", time.time()-start, "seconds")

fig.savefig("./pictures/explanation3.png")
