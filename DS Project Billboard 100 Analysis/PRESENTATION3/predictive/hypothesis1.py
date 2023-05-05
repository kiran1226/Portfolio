import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


start = time.time()
print("starting hypothesis1")

x = [i/150 for i in range(0, 150)]
y = [100+100*(p*np.log(p) + (1-p)*np.log(1-p)) for p in x]

print("finished hypothesis1 in", time.time()-start, "seconds")



g = sns.lineplot(x=[p*20 for p in x], y=y)


g.set(xlabel='hours of playtime until week #n', ylabel='rank in week #n+1', title=' Hypothesis')

g.set_xticks([2*i-1 for i in range(1,11)])
plt.gca().invert_yaxis()
fig = g.get_figure()
fig.savefig("./pictures/hypothesis1.png", bbox_inches="tight")
