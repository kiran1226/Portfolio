import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == '__main__':
    df=pd.read_csv("charts.csv")

    df['song-artist']=df['song']+" "+df['artist']
    df=df[['song-artist','peak-rank','weeks-on-board']]
    df=df.drop_duplicates('song-artist')
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    sns.set(font_scale=1.5)
    plt.figure()
    g = sns.scatterplot(x="weeks-on-board", y="peak-rank", data=df, palette="Blues", alpha=1  )
    g.set(xlabel='Weeks on Board', ylabel='Peak Ranking')
    plt.show()




