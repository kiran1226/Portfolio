import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from params import *



start = time.time()
print("starting explanation1")
a = -np.log(ratio)
b = -a*np.exp(100*a) / (1 - np.exp(99*a))

sns.set(font_scale=1.8)
g = sns.barplot(x=[i for i in range(1,101)], y=[(b*np.exp(-a*i)*100) for i in range(1,101)],color="salmon")
avg = np.average([(b*np.exp(-a*i)*100) for i in range(1,101)])



print("finished explanation1 in", time.time()-start, "seconds")



sns.lineplot(x=[i for i in range(100)], y=[avg for i in range(1,101)])
plt.plot(30, avg+0.1, 'k', marker=f"$mean={avg:.2f}$")


g.set(xlabel='Rank', ylabel='% of Plays', title=' ASSUMPTION: Exposure distribution over rank')

g.set_xticks([0]+[10*i-1 for i in range(1,11)])

fig = g.get_figure()
fig.savefig("./pictures/explanation1.png", bbox_inches="tight")
