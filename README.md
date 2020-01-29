# Spending Trends in Football Clubs and Prediction Model for Football Transfers.
Understanding Trends in Football Transfers and trying to build a prediction model to predict the market value of players using Python.

There are two parts to this project of mine. The first one is understanding the trend in football transfers over 14 seasons. The second part is where I try to build a prediction model to predict the market value of players in the future.

#### Some important things to consider before going into in:
- This is not the finalised version. As the weeks go by and as I learn new skills, I will keep adding to this repository. I will keep tweaking the code, playing with the model, adding more feature and updating this documentation as frequently as I can.
- The dataset I'll be using here is from Kaggle. Click [here](https://www.kaggle.com/vardan95ghazaryan/top-250-football-transfers-from-2000-to-2018) for the Kaggle link. And I'd like to thank Vardan for creating this excellent dataset.
- I would love it if you - the reader, have any comments or suggestions about anything. I'm a student and I intend to be one even after I graduate.
- The code is all in Python 3.7 and I use _PyCharm Community Edition_ as my IDE.

_Now let's dive in!_

# Overview

Football clubs these days are paying exorbitant prices to get their players. Defenders are going for €60-70 million, playmakers for €100-120 million and superstars for even more. Every club and fan will try and justify the amount spent on certain players and the points they make might be valid at the end. But all in all, paying millions in the 60s and 70s is too much and it's hurting the market in a bad way.

Harry Maguire might be the answer to your defensive problems, but paying €93 million for a player who's shown only glimpses of skill and has been good only over two seasons is just throwing money. But to be fair to Harry, he needs time to settle and United aren't in a good place right now. Maybe if the things turn around for them, we can see if Maguire is worth the money.

As the years go by, we can only expect the trend to rise - clubs will only go out of their way and we can see them paying more and more in the hopes of getting that one player who’ll take them to great heights. 

Rumours of PSG wonderkid, Kylian Mbappé, having a price tag of €500 million isn’t far from the truth. And only one club is even remotely considering to buy him at that price.

_Now how is that fair to the rest of the clubs?_

As I've mentioned before, the trend of paying more for a player will only increase as the years go by.

**Part I Goal:** Find the trend of how much money clubs splash for getting players. I'll try and explain the trend for which seasons saw a massive increase or decrease in spending; meaning, I'll try and attribute real-world stories and news to understand why there was an increase/decrease.

**Part II Goal:** Next part consists of creating a prediction model. Using regressive models, we can try and predict the market value of players in the coming seasons.

## About the Dataset:

This dataset contains all the transfers that have taken place from the 2000/2001 season till the 2018/2019 season.

It contains:
- Name of the Players
- Primary Position they play 
- Age of the Player
- The club they've been bought from
- The league they've been bought from
- The club that bought them
- The league their new club belongs to
- The season in which the transfer took place
- Market Value of the player
- Transfer Fee paid for the player

Here is a screengrab of the dataset as viewed on Kaggle.

![Dataset_Snap](https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Dataset_Snap.PNG)

#### Some things to know about the Dataset:
- Market Value for players in the seasons 2000/2001, 2001/2002, 2002/2003, 2003/2004 and a few in the other seasons are unavailable. So we'll be dropping them in our analysis.
- Since there is only limited data in the 2018/2019 season, **for Part I of the project, I'll be dropping them.**

# Part I

Before we go into the, let me explain a few parameters we'll be using extensively from the Dataset.

**Market Value:** The Market Value is the value that is set for a player based on his nationality, skill, position, age, brand value and prestige. Usually, young attacking players from the South American region and slick and skilled playmakers from the top European countries are valued highly compared to the others.

Players like Neymar, Vinicius Junior, Dybala, Eden Hazard, Marco Reus, and Kevin de Bruyne have a huge market value. Not only are they talented and proven, but they also bring a huge brand image with them making them the player to watch wherever they go.

**Transfer Fee:** The Transfer Fee is the amount of money a club pays to acquire the services of a player. A club can match the Market Value, pay more or even pay less for the player. It depends a lot on external factors. For starters, if a football club needs a goalkeeper badly and the transfer window deadline is fast approaching, they would be forced to pay more than the normal/market value - _the best example is Chelsea getting Kepa Arrizabalaga for a staggering €80 million making him the world's most expensive goalkeeper._

So, my goal here is to understand the trend of how clubs pay for players over multiple seasons. And to understand and study that, I've to introduce a new parameter; **Difference**.

                                            Difference = Tranfer Fee - Market Value

#### What does _Difference_ tell us?

It's simple enough to see what Difference is; it's just the difference between the Transfer Fee and Market Value. But in a footballing context, it's a bit more elaborate. As you know, a club sets a market value for a player based on various factors(age, position, birth country, skill, etc), and a buying club almost never pays the exact amount set by the selling club. It's sometimes(mostly) more than the market value and sometimes less.

With Difference, we can see how much more/less a club has paid for a certain player. And for the majority of the players, there is always a story or reason as to why the club paid what they paid. For example, as mentioned before, Chelsea needed a new keeper after the Thibaut Courtois left to Real Madrid, so they ended up paying a large amount for Arrizabalaga. Likewise, a few seasons earlier, Liverpool were in dire need of a defender and they ended up paying €78.8 million. These are some of the cases where the club paid way more than the market value.

There are also cases where the buying club paid less than the market value. A small example of this is when legendary journeyman Zlatan Ibrahimović went for a fraction less than his Market Value from Juventus to Inter. There is a reason for this which will be explained later.

This is what my simple named parameter, Difference tells us. As much as I'd love to attribute a reason to all the transfers that have a difference, it makes no sense to do it because not every transfer has a story and even if that was the case, I won't have the time to do it. _I'd be writing a magnum opus if I did that!_

Therefore, in an attempt to explain and paint the full picture of transfer trends, we find out the **Average Difference** over 14 seasons that football clubs have been paying. This is where I'd go about explaining why a certain season saw a rise/fall in spending.

Conclusively, the goal of this part is to study the trends of how football clubs have started to pay more and more for players in the past few seasons.

_We can argue on and on about how the market today is bloated and try to justify your club's spending, but we can all agree that €222 million for Neymar is an absurd amount of money._

Now to go into the code and explain a bit of everything I did. 

First, I import the necessary packages, read the CSV file and drop the rows with NaN market values. Then, I split the dataset by each season and find the _Difference for each player_ and using numpy we calculate the _Average Difference for that season._


```python
#Reading the dataset into an object
a = pd.read_csv('https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20I/Transfers.csv')

#Cleaning the Dataset of NaN values
a1 = a.dropna()

#Grouping the dataset by each Season
a2 = a1.groupby('Season')

#Find the difference for each player in 2004/2005 season
a3 = a2.get_group('2004-2005')
a3['Difference'] = a3['Transfer_fee'] - a3['Market_value']
#Finding the mean of this difference for the season
mean3 = round(np.mean(a3.Difference))
```

The above snippet is just for the 2004/2005 season; we do this for all the 14 seasons. Then we save all these Average Difference values in a separate CSV file called _Mean_Table.csv_. This table is going to help us plot the line graph for our analysis purpose.

This is how the _Mean_Table.csv_ table looks like.

Season | Average Difference
:---:  | :---:
2004/2005 | 258814
2005/2006 | 10822
2006/2007 | -298755
2007/2008 | 1651277
2008/2009 | 1522716
2009/2010 | 1532672
2010/2011 | 826929
2011/2012 | 131774
2012/2013 | -103333
2013/2014 | 1599980
2014/2015 | 2298612
2015/2016 | 4167077
2016/2017 | 5570964
2017/2018 | 6897261

From the table, you can see the values of Average Difference vary drastically between seasons than others with some even going into the negatives. Now, using matplot, I will plot the line graph to get a clear picture.

```python
#Reading the data from Mean_Table into an object
b = pd.read_csv('https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20I/Mean_Table.csv')

#Plotting a line graph for the data
b.plot(x='Season', y='Average Difference')
plt.title('The Average Difference Over All Seasons')
plt.xlabel('Season')
plt.ylabel('Average Difference')
plt.show()
```

And this is what the plot looks like:

![LineGraph_Snap](https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20I/LineGraph_Snap.PNG)

From the graph, we can see the Average Difference throughout the 14 seasons. I've marked out certain edges of prominent increase and decrease with red circles that I'll go about explaining.

## Circle A:

                                                  2005/2006 ----> 2006/2007
                                                  €10822    ----> -€298755

At circle A, we can see a decrease from ten thousand euros to an abysmal negative €3 million. The negative value is an indication of football clubs paying less than the market value for players in the 2006/2007 season. This means that on an average, players were being undersold.

This phenomenon can be attributed to the [_2006 Italian Football Scandal_](https://www.bbc.com/sport/football/49910626), a.k.a _Calciopoli_.

Towards the end of the 2005/2006 season, many Italian clubs were caught in a match-fixing scandal. Investigations discovered that teams paid money to get favourable referees for their matches helping them win games.

The Calciopoli scandal verdict was huge. Most notably, Italian Football Giants; _Juventus_ were fined €75,000, stripped of their 2004/2005 Serie A title, relegated to Serie B and deducted 30 points at the start of the 2006/2007 season. This relegation resulted in the majority of their big players leaving. And due to the scandal, Juventus didn't have any selling power over their players who didn't want to play in a lower league. They had to undersell big players like Zlatan Ibrahimović, Fabio Cannavaro, Patrick Vieira and Gianluca Zambrotta.

But how are few players being sold for less than their market value result in a **2860% decrease in spending**?

We have to remember that Juventus weren't the only club hurt by the Calciopoli scandal. Other clubs like A.C. Milan, Fiorentina, Lazio and Reggina were caught in the scandal. While A.C Milan and Reggina had points deducted and fines were given, Lazio and Fiorentina were relegated alongside Juventus. In the wake of the scandal, there was a mass exodus of players leaving the Serie A. Around 30 players who featured in the 2006 FIFA World Cup left. Despite Italy being champions, the scandal had hurt Serie A massively. Italian clubs were forced to sell players below the market value. This is the major reason for underspending by football clubs during this period.

## Circle B to Circle C:

                                                  2012/2013 ----> 2013/2014
                                                  -€10333   ----> €1599980

Back in 2011, the State of Qatar acquired Paris Saint-Germain F.C.(PSG) through _Qatar Sports Investments(QSI)_. But initially, the new owners didn't spend money. During and after the 2012/2013, PSG hit the market buying superstars like Zlatan Ibrahimović, Edinson Cavani, Thiago Silva and other talented players like Lavezzi, Marco Verratti, Lucas Moura and Digne.

Contrary to popular belief, PSG didn't splash huge amounts of money on these players. For the majority of their transfers, they paid only a fraction above the market value, they acquired Ibrahimović for **less than 43% of his market value**. Unlike today, PSG at the beginning of the takeover did good business and made many sensible buys ushering in their era of dominance over the Ligue One.

We can see a spike from negative ten thousand euros to a staggering €1.6 million in the season 2013/2014 and the reason may lie in one transfer; Gareth Bale moving from Tottenham Hotspurs to Real Madrid for €101 million. The _Los Blancos_ paid **55% more than the market value** for Bale. In my personal opinion, this I believe is where the trend of overpaying for remotely good players in the hopes that they'd become superstars began.

Don't get me wrong. Gareth Bale is a fantastic player and what he's done with Real Madrid is fantastic. But the amount of the money they've spent on him, the medical expenses and the negative-_ish_ PR he brings with him just doesn't justify his price tag. Every other big transfer after this only made things worse for the transfer market, in general expect, a few.

## Circle C to Circle D:


                         2013/2014 ---> 2014/2015 ----> 2015/2016 ----> 2016/2017 ----> 2017/2018
                          €1599980 ---> €2298612  ----> €4167077  ----> €5570964  ----> €6897261

Today's transfer market is bloated and it will only become worse as the seasons go by. And this can be seen in the meteoric rise the Average Difference makes from the 2014/2015 season to the 2017/2018 season; **from €1.6 million to €7 million**. This means that the average amount of money football clubs are paying for players above the market was €7 million as of 2017/2018. That is a massive **337% increase in overspending**. If this trend is to continue, which I'm sure it will, we could see huge amounts of money being paid by clubs to get talent.

Just to paint a clearer picture, here are a few transfers with huge overspending percentages in the five seasons:
- Neymar - €222 million, **122% more** than market value
- Virgil Van Dijk - €78.8 million, **162% more** than market value
- Paul Pogba - €105 million, **50% more** than the market value
- Ousmane Dembele - €115 million, **248% more** than the market value
- Anthony Martial - €60 million, **650% more** than the market value
- N’Golo Kante - €9 million, **100% more** than the market value

These are just a few examples of paying too much for a player. Hundreds of transfers over the past five years have been bloated and I can see it only increasing.

The takeover of clubs by oil money has completely changed the whole game. Pep Guardiola is given a war chest every window and gets every player he wants. Today, City can field two completely different squads that can challenge for the league title. Like attackers, today wing-backs and fullbacks are of high demand. This is due to the tactics of the game evolving frequently. Wing play has become essential in the past few seasons and the transfers show for it.

***Updates: 12/10/2019***
A quick update. I've applied pairplots using seaborn on the age of the player, market value and transfer fee.
Here is the code snippet:

```python
#Plotting the data in pairplots using seaborn
pp = sns.pairplot(d1, y_vars=['Age'], x_vars=['Market_value', 'Transfer_fee'], hue="Season")
plt.show()
```
And this is the output.

![PairPlot_Transfer](https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20I/PairPlot_Snap.png)

Due to the high number of entries, the plot looks a little messy. But I've circled out the portion that's interesting with a red circle. As you can see, players between the age of 20 to 25 are of valued higher in the market value irrespetive of which season. Young, skilled players always carry a huge valuation with them and these days, that can led to hype and high transfer fees. The yellow and blue circle indicates the market value of Mbappe and Transfer fee of Neymar respectively; the highest valued player and the costliest player.

## Conclusion:

Straight off, this is bad for the sport. As the bigger and richer clubs continue to pay exaggerated prices for players, the clubs without the funds and resources can be left behind without the means to compete. This creates a huge gap in the league tables and will see a huge decline in the overall quality of football each season.

Only a few footballers live up to their price tag. Players like Cristiano Ronaldo, Zinedine Zidane, and most recently Virgil Van Dijk are some of the best examples of players worth the extra money.

I don't see a straightforward solution to this problem. Yes...football associations across the globe do have spending caps and regulations to battle this, but none of them seems to be working as every transfer window we witness a big-money transfer that makes no sense at all.

Till the day football clubs stop paying way above the market value for players and pay for what they're worth and what they could bring to the club, instead of paying for hype and attention, it'll only become worse. But my pessimistic side says that it's never going to happen.


# Part II

In Part II, I'll try building a prediction model for the market value of players based on the data from the dataset. But first...

#### What is Linear Regression?
Linear Regression is a type of analysis which creates a model between two variables; the first one being the explanatory variable and the second one being the dependent variable.

Here, we are going to take **Market Value** as the explanatory variable because that's is what we need to calculate and **Transfer Fee** as the dependent variable.

> Note: Later on, I'll add more features to the dependent variable and try and create a well-defined model that what is present here.

#### Why predict Market Value and not Transfer Fee?
The transfer amount a club pays has a lot of outside factors influencing it, like the need for the player, the club's stature, their current form, their financial situation and many more. It's also influenced by the player, he should be willing to move and that brings in even more factors into the equation like personal ones such as willing to move to a new country or city and such. Sometimes players can force a move to the club they desire and the selling club could be forced to sell below the market value. Sometimes the buying club could be desperate for a player and the selling club can use this to their advantage and make the former pay more. 

These are just a few reasons why we can't truly predict the Transfer Fee. But what we can and will try to do is to predict the market value of future players.

We're going to be using _scikit learn_ to help us with building this prediction model. Like in Part I, I start by reading the dataset into an object and dropping the rows with NaN values.

I assign the Transfer Value to the dependent variable **y** and Market Value to the explanatory variable **X**. Then, using scikit learn, I split the data into training data and test data.

```python
c = pd.read_csv('https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20II/Transfers.csv')

c1 = c.dropna()

#Creating your Linear Regression objects; the dependant variable and explanatory variable
X = c1['Transfer_fee'].values.reshape(-1, 1)
y = c1['Market_value'].values.reshape(-1, 1)

#Splitting your data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

We use the training data to train the regression model and the test data to test the accuracy of this model.

Next, we create an object for Linear Regression and go on to fit the model and calculate the intercepts and slope.

```python
#Using Linear Regression from scikit learn, fitting the model and calculating the intercepts and slope
cx = LinearRegression()
cx.fit(X_train, y_train)
print('Intercept:', cx.intercept_)
print('Coefficient:', cx.coef_)
```

Finally, I use the test data to predict the values and plot a scatter graph with the linear regression line.

```python
#Predicting using test data
y_pred = cx.predict(X_test)

#Writing the actual and predicted values into a csv file
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.to_csv('Actual_Predicted.csv')

#Plotting the scatter plot and linear regression line
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=1)
plt.show()
```

![LinearPlot_Snap](https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20II/LinearPlot_Snap.PNG)

Now that we've trained and tested the model resulting in a table; _Actual_Predicted.csv_ table, and plotted the data in a scatter graph with Linear Regression, we can evaluate our model. 

To do that, we need to calculate three evaluation metrics:
- _Mean Absolute Error(MAE):_ This is the mean of the absolute value of errors.
- _Mean Squared Error(MSE):_ This is the mean of the squared errors.
- _Root Mean Squared Error(RMSE):_ This is the square root of the mean squared errors.

Again using scikit learn, we can find these values:

```python
mean = c1["Market_value"].mean()
print('Mean of Market Value:', mean)

#Calclulating evaluation metrics to evalute your regression model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

And the values are:

```
Mean of Market Value: 8622469.476744186
Mean Absolute Error: 3544174.170565751
Mean Squared Error: 27095585273554.777
Root Mean Squared Error: 5205341.9939092165
```

From these values, we can see that the **RSME**(5205342) **is more than 10% of the mean value** of Market Value(8622469).

#### What does this tell us?
Since our RSME is more than 10% of the mean of Market Value, our model isn't very accurate, but can still make decent predictions.

## Conclusion:

#### Why is the model inaccurate?
- Poor (or) Not Enough Features: As of the first version of this project, I haven't added multiple features in our model. Remember when I said, that the Market Value of a football player is determined by his age, birth country, position, skill and so on. And since the only dependant feature we took here is Transfer Fee, it isn't sufficient.


> Note: In the future, I will add more features, try to vectorize the non-integer values and try and create an accurate model.

- The Ever-changing Game: Football happens to be the most popular sport and that makes it an ever-changing sport. For a few years, midfielders are important and then all of a sudden, defenders are the most sought after. One year, Spain is producing the best young talent, another year, Dutch youngsters are setting the stage on fire! Even if we managed to have a perfect working model right now, it would become obsolete in a few seasons.

#### So, how do you take all this as quantifiable features and use them in a model for a more accurate prediction?
As Part I of this project says, we can always see trends and understand why such and such happened. But, external factors play a major role in football transfers, even fans and media can hype and inflate a player's market value.

If there was a way to take into account all these factors and predict the market value, football clubs then wouldn't be running around throwing money like they do today. But that's not that case!

***Remember?! In a few months or years, Mbappé can be bought for €500 million...that’s half a billion!! Let that sink in!***
