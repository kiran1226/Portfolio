# Presentation 2

The second presentation follows more closely an argument-driven data analysis.

We begin by noticing our dataset has an important quality: it represents "Motivation to consume Songs" at the beginning of the week, then "Actually consumed songs" in the end[[1]](https://www.billboard.com/pro/billboard-changes-streaming-weighting-hot-100-billboard-200/).

You might notice this exactly fits our needed Data in **Popularity** and **Exposure**, which allows for [Trajectory Analysis](https://www.publichealth.columbia.edu/research/population-health-methods/trajectory-analysis) as our next mode of investigation. Let's define this properly:

A Songs **Trajectory** is a function (often called y), representing for each week since arrival on billboard t, exactly the position of that song on the board as rank.

The value of that function then, represents exactly exposure to that song in the last week. This is because of how billboard calculates positions on board. Also its sum from t=0 to any point T represents the total exposure to the song until this point.

Popularity is exactly the time delayed variant of this function term, we can identify the popularity at any point in time, by its billboard position in the next week. We can thus derive the following equation:

Popularity(T-1) = y(T)  
Exposure(T) = Sum_{t=0, T} y(t)

In this formulation our research question looks like this:

Popularity(T) ?= f(Exposure(T))
Is Popularity related to Exposure until this point, or conversely

y(T+1) = f(Exposure(T))

Can we predict the future Rank value just by the aggregate Exposure value.
Another interesting question may lie in the finite difference. For linear functions relating popularity and Exposure, this significantly eases the question:
Δy(T+1) = Δf(Exposure(T))
.       =  f(ΔExposure(T))
.       =  f(y(T))
We would expect even this analysis to find some relation to the actual data, even though quadratic relations can not be characterized by this model

This can be converted into an analysis in various ways. We begin, easily enough, data first, structure agnostic, by computing a Scatter graph of the two values in the equation to determine the shape of f.


Slides

0 Title                -  
1 Motivation           R  
2 Motivation           R  
3 Question Familiarity R  
4 Data                 R  
5 Critique             B  
6 analysis             Y  
7 analysis             Y  
8 discussion           B  x2 time  
9 pipelines            B  x1/2 time  
10 pipelines           Y  x1/2 time  
0 sources              -  
