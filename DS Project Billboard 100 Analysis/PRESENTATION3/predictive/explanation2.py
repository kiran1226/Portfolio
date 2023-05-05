import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from params import *


print("loading file for explanation2")
start = time.time()
df = pd.read_csv(f"/home/miana/Downloads/weather_berlin/charts.csv")
total = df.groupby(by=["song", "artist"])

print("finished loading file in", time.time()-start, "seconds")


a = -np.log(ratio)
b = -a*np.exp(100*a) / (1 - np.exp(99*a))
T = int(len(total)*percentage_to_analyse)
print("starting explanation2")
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
    break

songs = pd.concat(songs).reset_index(drop=True)



ax = sns.lineplot(x='aggr', y=[songs["rank"].iloc[1]]*2, data=songs[1:3], alpha  = 1, palette = "Blues")
#ax.lines[0].set_linestyle("--")
ax.vlines(songs["aggr"].iloc[2], songs["rank"].iloc[1], songs["rank"].iloc[2], linestyle="dashed")

sns.lineplot(x='aggr', y='rank', data=songs, alpha  = 1, palette = "Blues", color=(3/255, 67/255, 223/255))
sns.scatterplot(x='aggr', y='rank', data=songs, alpha  = 1, palette = "Blues")

ax.set(xlabel='Exposure(T) in % of week plays', ylabel='Rank(T+1)', title='example song: (\"B\" Girls, Young And Restless)')

print("finished explanation2 in", time.time()-start, "seconds")

plt.gca().invert_yaxis()
fig = ax.get_figure()
fig.savefig("./pictures/explanation2.png")
