#Commucation
while our analysis is pretty cool we have made some issues in communication and that should now be alleviated.


Our core idea is the following: the rank values can be calculated into proper exposure values by assuming some distribution laid on top of the top100

So the amount of exposure one song gets is displayed as percentage of the total plays the billboard made during that week.

You can see the distribution we chose in file explanation1.png

If we use this distribution we can evaluate the core question of this entire project: how does exposure relate to liking?

How we calculate exposure you've just seen, so now as a quick aside: how do we operationalize liking?

Since Billboard records all streams in a week, we can be confident that the rank value at the end of the week corresponds to the motivation people had to stream the song over that time. We use "motivation to stream" as analogue for popularity, and then operationalize popularity at time t to be the same as its rank value in the next week, at time t+1.

Computationally, we need to take a song, sum over the distribution values up until some time t, and then look into the future to its rank value at time t+1. The relation between those two points, especially if we look at a lot of those, should reveal our effect. For example of a single song under this analysis, look at explanation2.png

The full analysis can be found under viz1.png

# Song Learnrate parameter

this leaves some stuff to be desired though. See how the later exposure values get dominated by single songs.

In the Bornstein analysis, stimulus complexity deserves an important role in moderating the strength and velocity of MEE[2].

We have subsequently found that our analysis fails in small samplesizes, because songs seem to react differently to a high exposure environment[3]

We try to use the former fact to alleviate problems with our analysis, by assuming the relationship between song and exposure is governed by another parameter.

From our second analysis:
Popularity(T-1) = y(T)   
Exposure(T) = Sum_{t=0, T} E_R(y(t))  
&  
y(T+1) = f(Exposure(T))  
which we will change to
y(T+1) = f(Exposure(T)*a)  

This means that Songs with a high learnrate paramter will take little exposure to rise (and fall) a lot in rank.

We thus define a as the average absolute steepness of the trajectory of each of the songs and see where that leads us.
=======================

After being done with this analysis we notice that in fact we cannot simply group songs by this factor, and it is my opinion that this is caused by songs not following a simple quadratic on different timescales, but instead following an entropy curve on different total information amounts.

An analysis of different song shapes (explanation3.png) seems to justify this, since the edges of our trajectory are seemingly much sharper than a quadratic would allow, making [this](https://de.wikipedia.org/wiki/Entropie_(Informationstheorie)#/media/Datei:Entropy_max.png) a more likely candidate for song trajectory base function

The graph presented there is -(log2(p)p + log2(1-p)(1-p)), going from 0 (total lack of knowledge of the song) to 1 (perfect representation of the song). Since by assumption (see bornstein, moderating factors: stimulus complexity) each song holds a different amount of total information, the percentage of known information about one song may increase at the differing rates as shown in the graph (viz1.png)

This seems like a complex step to program so I await feedback before implementing a normalization scheme based on that logic
