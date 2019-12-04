# Predictive Model for Football Transfers.
Understanding Trends in Football Transfers and trying to build a prediction model to predict the market value of players using Python.

There are two parts to this project of mine. The first one is understanding the trend in football transfers over 14 seasons. The second part is where I try to build a prediction model to try and predict the market value of players in the future.

#### Some important things to consider before going into in:
- This is not the finalised version. As the weeks go by and as I learn new skills, I will keep adding to this repository. I will keep tweaking the code, playing with the model, adding more feature and updating this documentation as frequent as I can.
- The dataset I'll be using here is from Kaggle. Click [here](https://www.kaggle.com/vardan95ghazaryan/top-250-football-transfers-from-2000-to-2018) for the Kaggle link. And I'd like to thank Vardan; for creating this excellent dataset.
- I would love it if you - the reader, has any comments or suggestions about anything. I'm a student and I intend to be one even after I graduate.
- The code is all Python 3.7 and I use PyCharm Community Edition as my IDE.

_Now let's dive in!_

# Overview

Football clubs these days are paying exorbitant prices to get their players. Defenders are going for €60-70 million, playmakers for €100-120 million and superstars for even more. Every club and fan will try and justify the amount spent on certain players and their the points they make might be valid at the end. But all in all, paying millions in the 60s and 70s is too much and it's hurting the market in a bad way.

Harry Maguire might be the answer to your defensive problems, but paying €93 million for a player who's shown only glimpses of skill and has been good only over two seasons is just throwing money. But to be fair to Harry, he needs to time to settle and United aren't in a good place right now. Maybe if the things turn around for them, we can see if Maguire is really worth the money.

As the years go by, we can only expect the trend to rise - clubs will only go out of their way and we can see them paying more and more in the hopes of getting that one player who’ll take them to great heights. 

Rumours of PSG wonderkid, Kylian Mbappé having a price tag of €500 million isn’t far from the truth. And only one club is even remotely considering to buy him at that price.

_Now how is that fair for the rest of the clubs?_

As I've mentioned before, the trend of paying more for a player will only increase as the years go by.

**Part I Goal:** Find the trend of how much money clubs splash for getting players. I'll try and explain the trend for which seasons saw an massive increase or decrease in spending; meaning, I'll try and attribute real-world stories and news to understand why there was an increase/decrease.

**Part II Goal:** Next part consists of creating a prediction model. Using regressive models, we can try and predict the market value of players in the coming seasons.

## About the Dataset:

This dataset contains all the transfers that have taken place from the 2000/2001 season till the 2018/2019 season.

It contains:
- Name of the PLayers
- Primary Position they play 
- Age of the Player
- The club they've been bought from
- The league they've been bought from
- The club that bought them
- The league their new club belongs to
- The season in which the transfer took place
- Market Value of the player
- Transfer Fee paid for the player

#### Some things to know about the Dataset:
- Market Value for players in the seasons 2000/2001, 2001/2002, 2002/2003, 2003/2004 and a few in the other seasons are unavailable. So we'll be dropping them in our analysis.
- Since there is only limited data in the 2018/2019 season, **for Part I of the project, I'll be dropping them.**

# Part I

Before we go into the, let me explain a few parameters we'll be using extensively from the Dataset.

**Market Value:** The Market Value is the value that is set for a player based on his nationality, skill, position, age, brand value and prestige. Usually, young attacking players from the South American region and slick and skilled playmakers from the top European countries are valued highly compared to the others.

Players like Neymar, Vinicius Junior, Dybala, Eden Hazard, Marco Reus, and Kevin de Bruyne have a huge market value. Not only are they talented and proven, but they also bring a huge brand image with them making them the player to watch wherever they go.

**Transfer Fee:** The Transfer Fee is the amount of money a club pays to acquire the services of a player. A club can match the Market Value, pay more or even pay less for the player. It depends on a lot of external factors. For starters, if a football club needs a goalkeeper badly and the transfer window deadline is fast approaching, they would be forced to pay more than the normal/market value - _the best example is Chelsea getting Kepa Arrizabalaga for a staggering €80 million making him the world's most expensive goalkeeper._

So, my goal here is to understand the trend of how clubs pay for players over multiple seasons. And to understand and study that, I've to introduce a new parameter; **Difference**.

                                             Difference = Tranfer Fee - Market Value

#### What does _Difference_ tell us?

It's simple enought to see what Difference is; it's just the difference between the Transfer Fee and Market Value. But in a footballing context, it's a bit more elaborate. As you know, a club sets a market value for a players based on various factors(age, position, birth country, skill, etc), and a buying club almost always never pays the exact amount set by the selling club. It's sometimes(mostly) more than the market value and sometimes less.

With Difference, we can see how much extra/less a club has paid for a certain player. And for majority of the players, there is always a story or reason as to why the club paid what they paid. For example, as mentioned before, Chelsea needed a new keeper after the Thibaut Courtois left to Real Madrid, so they ended up paying a large amount for Arrizabalaga. Likewise, a few season earlier, Liverpool were in dire need of a defender and they ended up paying €84.65 million. These are some of the cases where the club paid way more than the market value.

There are also cases where the buying club paid less than the market value. A small example for this is when legendary journeyman Zlatan Ibrahimović went for a fraction less than his Market Value from Juventus to Inter. There is a reason for this which will be explained later.

This is what my simple named parameter, Difference tells us. As much as I'd love to attribute a reason to all the transfers that has a difference, it makes no sense to do it because not every transfer has a story and even if that was the case, I won't have the time to do it. _I'd be writing a magnum opus if I did that!_

Therefore, in an attempt to explain and paint the full picture of transfer trends, we find out the **Average Difference** over 14 seasons that football clubs have been paying. This is where I'd go about explaining why certain season saw a rise/fall in spending.

Conclusivley, the goal of this part is to study the trends of how football clubs have started to pay more and more for players in the past few seasons.

_We can argue on and on about how the market today is bloated and try to justify your club's spending, but we can all agree that €222 million for Neymar is an absurd amount of money._

Now to go into the code and explain a bit of everything I did. 

After importing the necessary packages, reading the csv file and dropping the rows with NaN market values, we split the dataset by each season and find the _Difference for each player_ and the _Average Difference for that season._

