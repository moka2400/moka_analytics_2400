import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import metrics

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import RegressionResultsWrapper

import glob

# Data from https://www.kaggle.com/datasets/saife245/english-premier-league

# INTRO
# In this work we try and predict the winner of the English Premier League by looking at results from previous seasons.
# We do this using an overly simplified and grossly naive approach

# PITFALLS AND ASSUMPTIONS
# We only consider teams that have appeared in every single season in the data, i.e. This leads to our first assumption.
# Assumption 1: All teams included have performed equally well against teams that have not appeared in all PL seasons we consider

# We pool all the data from the different seasons which mean that we essentially assume that no teams become better or worse (e.g. Newcastle, Aston Villa) between seasons
# Assumption 2: We don't consider that a team may have significantly increased or decreased its level throughout the seasons and going into the next one

# We only consider goal differences in each game. The model may overestimate high goalscoring teams and not consider points. Winning 38 games by 1-0 is no better than winning 1 game 38-0 and drawing the rest.
# Assumption 3: Only goaldifferences per single match count

# ... And many, many more assumptions.

# The result is generally a very poor model trying to fit data way in over its head.
# There is a strong correlation and multicollineratity.

# Define the columns to select and the football clubs to filter by
columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
clubs_to_include = [
    'Man City',
    'Liverpool',
    'Man United',
    'Chelsea',
    'Tottenham',
    'Arsenal',
    'Everton',
    'West Ham',
    'Leicester City',
    'Crystal Palace',
    'Brighton',
    'Newcastle'
]
file_paths = glob.glob('../../data/pl_matches/*.csv')

def load_data() -> pd.DataFrame:
    dfs = [
    pd.read_csv(file, sep=",", header=0, decimal=".", index_col=0)[columns]
    .query("HomeTeam in @clubs_to_include and AwayTeam in @clubs_to_include")
    for file in file_paths]

    # Concatenate all filtered dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df['Date'] = combined_df['Date'].apply(parse_date)

    return combined_df

def regression_model(df: pd.DataFrame) -> RegressionResultsWrapper:
    # Sorted list of unique teams
    unique_teams = sorted(df['HomeTeam'].unique())

    # Goal difference for each game of similar dimension
    df['goal_differences'] = df['FTHG'] - df['FTAG']

    # Initializing design matrix for model
    design_matrix = pd.DataFrame(0, index=df.index, columns=unique_teams)

    for idx, row in df.iterrows():
        design_matrix.at[idx, row['HomeTeam']] = 1
        design_matrix.at[idx, row['AwayTeam']] = -1

    X = sm.add_constant(design_matrix)  # Add a constant for the intercept
    y = df['goal_differences'] # The dependant variable

    # Fit the OLS model
    model = sm.OLS(y, X).fit()

    return model

def plot_team_ratings(model):
    # Extract coefficients
    team_ratings = model.params[1:]  # Exclude the intercept

    # Create a DataFrame for plotting
    team_ratings_df = team_ratings.reset_index()
    team_ratings_df.columns = ['Team', 'Rating']

    # Sort by Rating for better visualization
    team_ratings_df = team_ratings_df.sort_values(by='Rating', ascending=False)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Team', y='Rating', data=team_ratings_df, color='blue', s=100)

    # Annotate team names
    for i in range(team_ratings_df.shape[0]):
        plt.text(i, team_ratings_df['Rating'].iloc[i], team_ratings_df['Team'].iloc[i], 
                 horizontalalignment='center', size='medium', color='black')

    plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
    plt.title('Team Ratings from Regression Model')
    plt.xlabel('Teams')
    plt.ylabel('Ratings')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

def predict_pl_winner():
    df = load_data()
    model = regression_model(df)

    print(model.summary())
    plot_team_ratings(model)

def parse_date(date_str):
    # Attempt to parse with 2-digit year format first
    try:
        return pd.to_datetime(date_str, format="%d/%m/%y")
    except ValueError:
        # If that fails, attempt with 4-digit year format
        return pd.to_datetime(date_str, format="%d/%m/%Y")

if __name__ == "__main__":
    predict_pl_winner()