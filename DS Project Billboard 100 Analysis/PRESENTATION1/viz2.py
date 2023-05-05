import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
def getYear(series):
    lst = []
    for element in series.date:
        lst.append(int(element[:4]))
    return pd.Series(lst)


def fit_root(x, y):
    a = 1
    b = 110
    c = y[0]
    E = lambda x, y: (np.sqrt(b-x)*a + c - y)**2
    for i in range(10000):
        da = 0
        db = 0
        dc = 0
        EGesamt = 0
        for x0,y0 in zip(x,y):
            E2 = 2 * (np.sqrt(b-x0)*a + c - y0)
            EGesamt += E(x0, y0)
            da +=  E2 * np.sqrt(b-x0)
            db +=  E2 * a / (2*np.sqrt(b-x0))
            dc +=  E2
        if i % 1000 == 0:
            print("E:", EGesamt)
        a -= da * 0.01 / len(y)
        b -= db * 0.01 / len(y)
        c -= dc * 0.01 / len(y)
    return a, b, c


df = pd.read_csv(f"/home/miana/Downloads/weather_berlin/charts.csv")
SongID = df[df["weeks-on-board"]==1][["song", "rank", "date", "artist"]]

total = SongID.merge(df.groupby(by=["song", "artist"]).max()["weeks-on-board"], how="inner", on=["song", "artist"]).assign(year=getYear )
print(total.head())


rank_axis = [*range(1, 101)]
median = [total.loc[total["rank"]==i]["weeks-on-board"].median() for i in range(1, 101)]
a,b,c = 1.187470473666365, 108.67155876612527, 3.8025608666670325 #  ==fit_root(rank_axis, median)
f = lambda x: np.sqrt(b-x)*a+c

sns.set_context("notebook")
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plt.figure()
plt.plot(median, rank_axis, 'ro', alpha=0.5)
plt.plot([f(x) for x in rank_axis], rank_axis, 'r-')
g = sns.scatterplot(x="weeks-on-board", y="rank", data=total,  palette="Blues", alpha=0.8)
g.set(xlabel='Timespan in top100', ylabel='Initial Ranking')
plt.show()
