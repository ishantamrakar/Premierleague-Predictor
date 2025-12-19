import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os

class PremierLeagueDataset(Dataset):
    def __init__(self, data, labels, home_team_ids, away_team_ids):
        self.data = data
        self.labels = labels
        self.home_team_ids = home_team_ids
        self.away_team_ids = away_team_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.home_team_ids[idx], self.away_team_ids[idx]

class PremierLeagueDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.team_to_id = {}
        self.id_to_team = {}
        self.num_teams = 0

    def _create_team_id_mapping(self):
        if self.raw_data is None:
            return

        unique_teams = pd.concat([self.raw_data['HomeTeam'], self.raw_data['AwayTeam']]).unique()
        self.team_to_id = {team: i for i, team in enumerate(sorted(unique_teams))}
        self.id_to_team = {i: team for team, i in self.team_to_id.items()}
        self.num_teams = len(unique_teams)

    def load_data(self):
        """Loads raw Premier League match data."""
        try:
            csv_path = os.path.join(self.data_path, 'dataset/2000-2025 team stats/epl_final.csv')
            self.raw_data = pd.read_csv(csv_path, encoding='latin1')
            self._create_team_id_mapping() # Create mapping after loading data
        except FileNotFoundError:
            print(f"Error: Data file not found at {csv_path}")
            self.raw_data = None

    def preprocess_data(self):
        """Preprocesses the raw data to calculate team form and prepare features."""
        if self.raw_data is None:
            return

        processed_df = self.raw_data.copy()

        processed_df['MatchDate'] = pd.to_datetime(processed_df['MatchDate'], dayfirst=True)
        processed_df.sort_values(by='MatchDate', inplace=True)

        result_mapping = {'A': 0, 'D': 1, 'H': 2}
        processed_df['FTR_numerical'] = processed_df['FullTimeResult'].map(result_mapping)
        
        home_points_map = {'H': 3, 'D': 1, 'A': 0}
        away_points_map = {'A': 3, 'D': 1, 'H': 0}
        processed_df['HomePoints'] = processed_df['FullTimeResult'].map(home_points_map)
        processed_df['AwayPoints'] = processed_df['FullTimeResult'].map(away_points_map)

        processed_df.dropna(subset=['FTR_numerical', 'HomePoints', 'AwayPoints'], inplace=True)
        
        # --- Pre-calculate Season Ranks and League Averages ---
        season_ranks = {}
        league_averages = {}
        unique_seasons = processed_df['Season'].unique()

        for season in unique_seasons:
            season_df = processed_df[processed_df['Season'] == season]
            
            league_averages[season] = {
                'avg_home_goals': season_df['FullTimeHomeGoals'].mean(),
                'avg_away_goals': season_df['FullTimeAwayGoals'].mean()
            }
            
            team_points = {}
            for team in season_df['HomeTeam'].unique():
                home_points = season_df[season_df['HomeTeam'] == team]['HomePoints'].sum()
                away_points = season_df[season_df['AwayTeam'] == team]['AwayPoints'].sum()
                team_points[team] = home_points + away_points
            
            ranked_teams = sorted(team_points.items(), key=lambda item: item[1], reverse=True)
            season_ranks[season] = {team: rank + 1 for rank, (team, points) in enumerate(ranked_teams)}

        # --- Feature Calculation Loop ---
        features = {
            'HomeTeam_Form': [], 'AwayTeam_Form': [], 'Home_Weighted_Form': [], 'Away_Weighted_Form': [],
            'Home_Attack_Strength': [], 'Home_Defense_Strength': [], 'Away_Attack_Strength': [], 'Away_Defense_Strength': [],
            'H2H_HomeWins_Pct': [], 'H2H_Draws_Pct': [], 'H2H_AwayWins_Pct': [], 'H2H_Home_AvgGoals': [], 'H2H_Away_AvgGoals': []
        }

        for index, row in processed_df.iterrows():
            home_team, away_team, match_date, season = row['HomeTeam'], row['AwayTeam'], row['MatchDate'], row['Season']
            
            # Past matches for form, strength, etc.
            home_team_matches = processed_df[((processed_df['HomeTeam'] == home_team) | (processed_df['AwayTeam'] == home_team)) & (processed_df['MatchDate'] < match_date)].tail(5)
            away_team_matches = processed_df[((processed_df['HomeTeam'] == away_team) | (processed_df['AwayTeam'] == away_team)) & (processed_df['MatchDate'] < match_date)].tail(5)
            
            # --- Form & Weighted Form ---
            home_form_points = home_team_matches.apply(lambda r: r['HomePoints'] if r['HomeTeam'] == home_team else r['AwayPoints'], axis=1)
            features['HomeTeam_Form'].append(home_form_points.sum())
            
            away_form_points = away_team_matches.apply(lambda r: r['HomePoints'] if r['HomeTeam'] == away_team else r['AwayPoints'], axis=1)
            features['AwayTeam_Form'].append(away_form_points.sum())
            
            home_weighted_form, away_weighted_form = 0, 0
            prev_season_str = f"{int(season.split('/')[0]) - 1}/{str(int(season.split('/')[1]) - 1).zfill(2)}"
            max_rank = len(season_ranks.get(prev_season_str, {}))

            for i, past_match in home_team_matches.iterrows():
                opponent = past_match['AwayTeam'] if past_match['HomeTeam'] == home_team else past_match['HomeTeam']
                opponent_rank = season_ranks.get(prev_season_str, {}).get(opponent, max_rank + 1)
                weight = (max_rank - opponent_rank + 1) / max_rank if max_rank > 0 else 1
                home_weighted_form += home_form_points.loc[i] * weight
            features['Home_Weighted_Form'].append(home_weighted_form)

            for i, past_match in away_team_matches.iterrows():
                opponent = past_match['AwayTeam'] if past_match['HomeTeam'] == away_team else past_match['HomeTeam']
                opponent_rank = season_ranks.get(prev_season_str, {}).get(opponent, max_rank + 1)
                weight = (max_rank - opponent_rank + 1) / max_rank if max_rank > 0 else 1
                away_weighted_form += away_form_points.loc[i] * weight
            features['Away_Weighted_Form'].append(away_weighted_form)
            
            # --- Offensive/Defensive Strength ---
            league_avgs = league_averages.get(season, {})
            avg_home_g, avg_away_g = league_avgs.get('avg_home_goals', 1), league_avgs.get('avg_away_goals', 1)

            home_scored = home_team_matches.apply(lambda r: r['FullTimeHomeGoals'] if r['HomeTeam'] == home_team else r['FullTimeAwayGoals'], axis=1).mean()
            home_conceded = home_team_matches.apply(lambda r: r['FullTimeAwayGoals'] if r['HomeTeam'] == home_team else r['FullTimeHomeGoals'], axis=1).mean()
            away_scored = away_team_matches.apply(lambda r: r['FullTimeHomeGoals'] if r['HomeTeam'] == away_team else r['FullTimeAwayGoals'], axis=1).mean()
            away_conceded = away_team_matches.apply(lambda r: r['FullTimeAwayGoals'] if r['HomeTeam'] == away_team else r['FullTimeHomeGoals'], axis=1).mean()

            features['Home_Attack_Strength'].append(home_scored / avg_home_g if avg_home_g > 0 and not np.isnan(home_scored) else 1)
            features['Home_Defense_Strength'].append(home_conceded / avg_away_g if avg_away_g > 0 and not np.isnan(home_conceded) else 1)
            features['Away_Attack_Strength'].append(away_scored / avg_away_g if avg_away_g > 0 and not np.isnan(away_scored) else 1)
            features['Away_Defense_Strength'].append(away_conceded / avg_home_g if avg_home_g > 0 and not np.isnan(away_conceded) else 1)

            # --- H2H Stats ---
            h2h_matches = processed_df[(((processed_df['HomeTeam'] == home_team) & (processed_df['AwayTeam'] == away_team)) | ((processed_df['HomeTeam'] == away_team) & (processed_df['AwayTeam'] == home_team))) & (processed_df['MatchDate'] < match_date)]
            if not h2h_matches.empty:
                h_wins = sum((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'H')) + sum((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'A'))
                a_wins = sum((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'H')) + sum((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'A'))
                draws = len(h2h_matches) - h_wins - a_wins
                
                # Check if goals lists are empty before taking mean
                h_goals_list = h2h_matches.apply(lambda r: r['FullTimeHomeGoals'] if r['HomeTeam'] == home_team else r['FullTimeAwayGoals'], axis=1)
                a_goals_list = h2h_matches.apply(lambda r: r['FullTimeAwayGoals'] if r['HomeTeam'] == home_team else r['FullTimeHomeGoals'], axis=1)

                h_goals_mean = h_goals_list.mean() if not h_goals_list.empty else 0
                a_goals_mean = a_goals_list.mean() if not a_goals_list.empty else 0
                
                features['H2H_HomeWins_Pct'].append(h_wins / len(h2h_matches))
                features['H2H_Draws_Pct'].append(draws / len(h2h_matches))
                features['H2H_AwayWins_Pct'].append(a_wins / len(h2h_matches))
                features['H2H_Home_AvgGoals'].append(h_goals_mean)
                features['H2H_Away_AvgGoals'].append(a_goals_mean)
            else:
                for k in ['H2H_HomeWins_Pct', 'H2H_Draws_Pct', 'H2H_AwayWins_Pct', 'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals']: features[k].append(0)

        for k, v in features.items():
            processed_df[k] = v
            
        processed_df['Home_Gameweek'] = processed_df.groupby(['Season', 'HomeTeam']).cumcount() + 1
        processed_df['Away_Gameweek'] = processed_df.groupby(['Season', 'AwayTeam']).cumcount() + 1
        processed_df['Match_of_the_Season'] = processed_df[['Home_Gameweek', 'Away_Gameweek']].max(axis=1)
        processed_df.drop(columns=['Home_Gameweek', 'Away_Gameweek'], inplace=True)
        
        # --- Map Team Names to IDs ---
        processed_df['HomeTeam_ID'] = processed_df['HomeTeam'].map(self.team_to_id)
        processed_df['AwayTeam_ID'] = processed_df['AwayTeam'].map(self.team_to_id)
        
        self.processed_data = processed_df

    def get_dataloaders(self, batch_size=32):
        if self.processed_data is None:
            return None, None

        feature_cols = [
            'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
            'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
            'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
            'HomeTeam_Form', 'AwayTeam_Form', 'Match_of_the_Season',
            'H2H_HomeWins_Pct', 'H2H_Draws_Pct', 'H2H_AwayWins_Pct', 'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals',
            'Home_Attack_Strength', 'Home_Defense_Strength', 'Away_Attack_Strength', 'Away_Defense_Strength',
            'Home_Weighted_Form', 'Away_Weighted_Form'
        ]
        
        existing_cols = [col for col in feature_cols if col in self.processed_data.columns]
        
        self.processed_data.dropna(subset=existing_cols + ['FTR_numerical', 'HomeTeam_ID', 'AwayTeam_ID'], inplace=True)

        X = self.processed_data[existing_cols].values
        y = self.processed_data['FTR_numerical'].values
        home_team_ids = self.processed_data['HomeTeam_ID'].values
        away_team_ids = self.processed_data['AwayTeam_ID'].values

        # --- Time-based Split ---
        seasons = sorted(self.processed_data['Season'].unique())
        latest_season = seasons[-1]
        
        train_indices = self.processed_data[self.processed_data['Season'] != latest_season].index
        test_indices = self.processed_data[self.processed_data['Season'] == latest_season].index

        X_train, y_train = X[train_indices], y[train_indices]
        home_team_ids_train, away_team_ids_train = home_team_ids[train_indices], away_team_ids[train_indices]

        X_test, y_test = X[test_indices], y[test_indices]
        home_team_ids_test, away_team_ids_test = home_team_ids[test_indices], away_team_ids[test_indices]


        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        home_team_ids_train_tensor = torch.tensor(home_team_ids_train, dtype=torch.long)
        away_team_ids_train_tensor = torch.tensor(away_team_ids_train, dtype=torch.long)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        home_team_ids_test_tensor = torch.tensor(home_team_ids_test, dtype=torch.long)
        away_team_ids_test_tensor = torch.tensor(away_team_ids_test, dtype=torch.long)

        train_dataset = PremierLeagueDataset(X_train_tensor, y_train_tensor, home_team_ids_train_tensor, away_team_ids_train_tensor)
        test_dataset = PremierLeagueDataset(X_test_tensor, y_test_tensor, home_team_ids_test_tensor, away_team_ids_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training on seasons before {latest_season}. Validating on season {latest_season}.")
        return train_loader, test_loader

if __name__ == '__main__':
    data_loader = PremierLeagueDataLoader(data_path='.')
    data_loader.load_data()
    if data_loader.raw_data is not None:
        data_loader.preprocess_data()
        if data_loader.processed_data is not None:
            print("--- Sample of Processed Data ---")
            print(data_loader.processed_data[[
                'HomeTeam', 'AwayTeam', 'Match_of_the_Season', 'HomeTeam_Form', 'AwayTeam_Form', 
                'Home_Weighted_Form', 'Away_Weighted_Form', 'Home_Attack_Strength', 'Away_Attack_Strength',
                'H2H_HomeWins_Pct', 'H2H_AwayWins_Pct'
            ]].sample(10))