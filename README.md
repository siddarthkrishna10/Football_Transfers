# Predictive Model for Football Transfers.
Understanding Trends in Football Transfers and trying to build a prediction model to predict the market value of players using Python.

There are two parts to this project of mine. The first one is understanding the trend in football transfers over 14 seasons. The second part is where I try to build a prediction model to try and predict the market value of players in the future.

#### Just some important things to consider before going into it:
- This is not the finalised version. As the weeks go by and as I learn new skills, I will keep adding to this repository. I will keep tweaking the code, playing with the model, adding more feature and updating this documentation as frequent as I can.
- The dataset I'll be using here is from Kaggle. Click [here](https://www.kaggle.com/vardan95ghazaryan/top-250-football-transfers-from-2000-to-2018) for the Kaggle link. And I'd like to thank Vardan; for creating this excellent dataset.
- I would love it if you - the reader, has any comments or suggestions about anything. I'm a student and I intend to be one even after I graduate.

## Here is the Technical README:

### Files for PART I:
- _Transfers.csv_ - The CSV file that contains the full Kaggle dataset.
- _Calculating_Difference.py_ - The python file that calculates the Difference metric and the Average Difference.
  - _Mean_Table.csv_ is created.
  - _Line_Graph.png_ - Image of the line graph
- _Plot_Difference.py_ - Python file where the line graph for Average Difference is plotted.

### Packages used in PART I:
- _import pandas as pd_
- _import numpy as np_
- _import matplotlib.pyplot as plt_

### Files for PART II:
- _Transfers.csv_ - The CSV file that contains the full Kaggle dataset.
- _Plot_Regression.py_ - Python file for creating the full Linear Regression model.
  - _Actual_Predicted.csv_ is created.
  - _Linear_Regression.png_ - Image of the linear regression model and scatter plot

### Packages used in PART II:
- _import numpy as np_
- _import pandas as pd_
- _import matplotlib.pyplot as plt_
- _from sklearn.model_selection import train_test_split_
- _from sklearn.linear_model import LinearRegression_
- _from sklearn import metrics_

The code is all Python 3.7 and I use PyCharm Community Edition as my IDE.

_Now let's dive in!_

# Overview

Football clubs these days are paying exorbitant prices to get their players. Defenders are going for €60-70 million, playmakers for €100-120 million and superstars for even more. Every club and fan will try and justify the amount spent on certain players and their the points they make might be valid at the end. But all in all, paying millions in the 60s and 70s is too much and it's hurting the market in a bad way.

Harry Maguire might be the answer to your defensive problems, but paying €93 million for a player who's shown only glimpses of skill and has been good only over two seasons is just throwing money. But to be fair to Harry, he needs to time to settle and United aren't in a good place right now. Maybe if the things turn around for them, we can see if Maguire is really worth the money.

As the years go by, we can only expect the trend to rise - clubs will only go out of their way and we can see them paying more and more in the hopes of getting that one player who’ll take them to great heights. 

Rumours of PSG wonderkid, Kylian Mbappé having a price tag of €500 million isn’t far from the truth. And only one club is even remotely considering to buy him at that price.

_Now how is that fair for the rest of the clubs?_
