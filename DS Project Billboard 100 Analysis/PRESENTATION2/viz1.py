import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


df = pd.read_csv(f"/home/miana/Downloads/weather_berlin/charts.csv")
SongID = df[df["weeks-on-board"]==1][["song", "rank", "date", "artist"]]

total = df.groupby(by=["song", "artist"])


songs = []
n = 0
for name, group in total:
    max_lifespan=group['weeks-on-board'].max()
    S = 0
    for i in range(len(group.index)):
        S += 100 - group.iloc[i]["rank"]
        group.iloc[i, group.columns.get_loc("weeks-on-board")]=group.iloc[i, group.columns.get_loc("weeks-on-board")]#/max_lifespan
    group["blue"] = [S for i in group.index]
    songs.append(group)
    n += 1
    if n==400:
        break
x=[]
sng=[]
songs = pd.concat(songs).reset_index(drop=True)
for i in range(10):
    p=i/10
    p2=(i+1)/10
    sng.append(songs.loc[(songs['weeks-on-board']>p) & (songs['weeks-on-board']<p2)]['rank'].mean())
    x.append(p+(1/20))

g=sns.lineplot(x='weeks-on-board', y='rank',hue="blue", data=songs,alpha=0.5)
g.set(xlabel='Timespan in Percentage')

sns.lineplot(y=sng, x=x,color='black',linewidth=5)
plt.gca().invert_yaxis()
plt.show()
