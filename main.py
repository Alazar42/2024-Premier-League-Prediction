import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Loading Dataset
data = pd.read_csv('datasets/PremierLeague.csv')

# Creating features for each team
features = []

teams = pd.concat([data['home_team_name'], data['away_team_name']]).unique()

for team in teams:
    team_data = data[(data['home_team_name'] == team) | (data['away_team_name'] == team)]
    wins = len(team_data[(team_data['home_team_name'] == team) & (team_data['home_team_goals'] > team_data['away_team_goals'])]) + \
           len(team_data[(team_data['away_team_name'] == team) & (team_data['away_team_goals'] > team_data['home_team_goals'])])
    draws = len(team_data[(team_data['home_team_name'] == team) & (team_data['home_team_goals'] == team_data['away_team_goals'])]) + \
            len(team_data[(team_data['away_team_name'] == team) & (team_data['away_team_goals'] == team_data['home_team_goals'])])
    losses = len(team_data[(team_data['home_team_name'] == team) & (team_data['home_team_goals'] < team_data['away_team_goals'])]) + \
             len(team_data[(team_data['away_team_name'] == team) & (team_data['away_team_goals'] < team_data['home_team_goals'])])
    avg_goals_scored = team_data[['home_team_goals', 'away_team_goals']].apply(lambda row: row.iloc[0] if row.iloc[0] > row.iloc[1] else row.iloc[1], axis=1).mean()
    avg_goals_conceded = team_data[['home_team_goals', 'away_team_goals']].apply(lambda row: row.iloc[1] if row.iloc[0] > row.iloc[1] else row.iloc[0], axis=1).mean()
    
    features.append({
        'team': team,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'avg_goals_scored': avg_goals_scored,
        'avg_goals_conceded': avg_goals_conceded,
    })

features_df = pd.DataFrame(features)

# Generate target variable based on standings
features_df['rank'] = features_df['wins'] * 3 + features_df['draws']  # Example scoring
features_df = features_df.sort_values(by='rank', ascending=False).reset_index(drop=True)
features_df['rank'] = features_df.index + 1  # Rank is 1-based

# Prepare data for training
X = features_df.drop(['team', 'rank'], axis=1)
y = features_df['rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Generate target variable based on standings
features_df['rank'] = features_df['wins'] * 3 + features_df['draws']  # Example scoring
features_df = features_df.sort_values(by='rank', ascending=False).reset_index(drop=True)
features_df['rank'] = features_df.index + 1  # Rank is 1-based

# Reverse the rank to show higher numbers as better standings
features_df['rank'] = features_df['rank'].max() - features_df['rank'] + 1

# Prepare for prediction
X = features_df.drop(['team', 'rank'], axis=1)
y = features_df['rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Make predictions
features_df['predicted_rank'] = model.predict(X)
features_df = features_df[['team', 'predicted_rank']].sort_values(by='predicted_rank', ascending=False).reset_index(drop=True)

print("Predicted Standings for Next Season:")
print(features_df)
