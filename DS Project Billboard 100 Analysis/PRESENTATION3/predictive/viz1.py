import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from params import *

print("loading file for viz1")
start = time.time()
df = pd.read_csv(f"/home/miana/Downloads/weather_berlin/charts.csv")
total = df.groupby(by=["song", "artist"])

print("finished loading file in", time.time()-start, "seconds")

a = -np.log(ratio)
b = -a*np.exp(100*a) / (1 - np.exp(99*a))
T = int(len(total)*percentage_to_analyse)
print("starting viz1")
start = time.time()
songs = []
n = 1
for name, group in total:
    group["date"] = pd.to_datetime(group["date"])
    group = group.sort_values("date")
    group["aggr"] = 0
    S = 0
    learnrate = 0
    for i in range(len(group.index)-1):
        learnrate += np.absolute(group.iloc[i+1, group.columns.get_loc("rank")]-group.iloc[i, group.columns.get_loc("rank")]) / (len(group.index)-1)
    for i in range(len(group.index)):
        group.iloc[i, group.columns.get_loc("aggr")] = S
        S += (b*np.exp(-a*group.iloc[i]["rank"])*100)#*learnrate
    group["color"] = learnrate
    songs.append(group)
    n += 1
    if n >= T:
        print("reached", T, "songs")
        break

songs = pd.concat(songs).reset_index(drop=True)
median = []
for i in range(int(max(songs["aggr"]))+1):
    if len(songs.loc[ (songs["aggr"]<(i+2)) & (songs["aggr"]>(i-2)) ]["rank"]) > 10 or i < 6:
        median.append(songs.loc[ (songs["aggr"]<(i+2)) & (songs["aggr"]>(i-2)) ]["rank"].median())
    else:
        break


g = sns.scatterplot(x='aggr', y='rank', data=songs,  alpha  = 0.7)
sns.lineplot(data = median, color="red")
g.set(xlabel='Exposure(T) in % of week plays', ylabel='Rank(T+1)')

print("finished viz1 in", time.time()-start, "seconds")

plt.gca().invert_yaxis()

fig = g.get_figure()
fig.savefig("./pictures/viz1.png")
