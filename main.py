import pandas as pd
import numpy as np
from sklearn import linear_model
import requests
from nba_api.stats import endpoints
from matplotlib import pyplot as plt

data = endpoints.leagueleaders.LeagueLeaders() # Access the league leaders module and assign the class to "data"
df = data.league_leaders.get_data_frame() # Create dataframe for our data

df.head() # Peek at first 5 rows (top 5 players)

x, y = df.FGA/df.GP, df.PTS/df.GP # Dividing each statistic by games played to find the per game avg
x = np.array(x).reshape(-1,1) # reshape array from 1D to 2D for linear model
y = np.array(y).reshape(-1,1)

model = linear_model.LinearRegression()
model.fit(x,y) # fit the modeling data using field goals attempted and points per game

r2 = round(model.score(x,y),2)
predicted_y = model.predict(x)

# graph visualization
plt.scatter(x, y, s=15, alpha=.5)
plt.plot(x, predicted_y, color='black')
plt.title('NBA - Relationship Between FGA and PPG')
plt.xlabel('FGA per Game')
plt.ylabel('Points Per Game')
plt.text(5, 20, f'R2={r2}')

# label for top 5 scoring players (does not necessarily translate to ppg leaders)

plt.annotate(df.PLAYER[0],
             (x[0], y[0]),
             (x[0] - 7, y[0] - 3.5),
             arrowprops=dict(arrowstyle='-'))

plt.annotate(df.PLAYER[1],
             (x[1], y[1]),
             (x[1] - 7, y[1]),
             arrowprops=dict(arrowstyle='-'))

plt.annotate(df.PLAYER[2],
             (x[2], y[2]),
             (x[2] - 9, y[2] - 1),
             arrowprops=dict(arrowstyle='-'))

plt.annotate(df.PLAYER[3],
             (x[3], y[3]),
             (x[3] - 12, y[3] - 1.5),
             arrowprops=dict(arrowstyle='-'))

plt.annotate(df.PLAYER[4],
             (x[4], y[4]),
             (x[4] - 10, y[4] - 3),
             arrowprops=dict(arrowstyle='-'))

plt.savefig('graph.png', dpi=300)