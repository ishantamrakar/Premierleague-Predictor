import pandas as pd
import numpy as np
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

    def load_data(self):
        try:
            csv_path = os.path.join(self.data_path, 'dataset/2000-2025 team stats/epl_final.csv')
            self.raw_data = pd.read_csv(csv_path, encoding='latin1')
            self._create_team_id_mapping()
        except FileNotFoundError:
            print(f"Error: Data file not found at {csv_path}")
            self.raw_data = None

    def _create_team_id_mapping(self):
        if self.raw_data is None: return
        unique_teams = pd.concat([self.raw_data['HomeTeam'], self.raw_data['AwayTeam']]).unique()
        self.team_to_id = {team: i for i, team in enumerate(sorted(unique_teams))}
        self.id_to_team = {i: team for team, i in self.team_to_id.items()}
        self.num_teams = len(unique_teams)

    def preprocess_data(self):
        if self.raw_data is None: return

        processed_df = self.raw_data.copy()
        processed_df['MatchDate'] = pd.to_datetime(processed_df['MatchDate'], dayfirst=True, errors='coerce')
        processed_df.dropna(subset=['MatchDate'], inplace=True)
        processed_df.sort_values(by='MatchDate', inplace=True)
        processed_df.reset_index(drop=True, inplace=True)

        result_mapping = {'A': 0, 'D': 1, 'H': 2}
        processed_df['FTR_numerical'] = processed_df['FullTimeResult'].map(result_mapping)
        home_points_map = {'H': 3, 'D': 1, 'A': 0}
        away_points_map = {'A': 3, 'D': 1, 'H': 0}
        processed_df['HomePoints'] = processed_df['FullTimeResult'].map(home_points_map)
        processed_df['AwayPoints'] = processed_df['FullTimeResult'].map(away_points_map)
        
        # Pre-calculate season-wide stats
        season_ranks, league_averages = self._precompute_stats(processed_df)

        # Calculate features iteratively
        feature_lists = self._calculate_features(processed_df, season_ranks, league_averages)
        for name, data in feature_lists.items():
            processed_df[name] = data
            
        # Gameweek
        processed_df['Home_Gameweek'] = processed_df.groupby(['Season', 'HomeTeam']).cumcount() + 1
        processed_df['Away_Gameweek'] = processed_df.groupby(['Season', 'AwayTeam']).cumcount() + 1
        processed_df['Match_of_the_Season'] = processed_df[['Home_Gameweek', 'Away_Gameweek']].max(axis=1)
        processed_df.drop(columns=['Home_Gameweek', 'Away_Gameweek'], inplace=True)

        processed_df['HomeTeam_ID'] = processed_df['HomeTeam'].map(self.team_to_id)
        processed_df['AwayTeam_ID'] = processed_df['AwayTeam'].map(self.team_to_id)
        
        self.processed_data = processed_df

    def _precompute_stats(self, df):
        season_ranks, league_averages = {}, {}
        unique_seasons = df['Season'].unique()
        for season in unique_seasons:
            season_df = df[df['Season'] == season]
            league_averages[season] = {'avg_home_goals': season_df['FullTimeHomeGoals'].mean(), 'avg_away_goals': season_df['FullTimeAwayGoals'].mean()}
            team_points = {team: (season_df.loc[season_df['HomeTeam'] == team, 'HomePoints'].sum() + season_df.loc[season_df['AwayTeam'] == team, 'AwayPoints'].sum()) for team in pd.concat([season_df['HomeTeam'], season_df['AwayTeam']]).unique()}
            ranked_teams = sorted(team_points.items(), key=lambda item: item[1], reverse=True)
            season_ranks[season] = {team: rank + 1 for rank, (team, points) in enumerate(ranked_teams)}
        return season_ranks, league_averages

    def _calculate_features(self, df, season_ranks, league_averages):
        feature_lists = { 'HomeTeam_Form': [], 'AwayTeam_Form': [], 'Home_Weighted_Form': [], 'Away_Weighted_Form': [], 'Home_Attack_Strength': [], 'Home_Defense_Strength': [], 'Away_Attack_Strength': [], 'Away_Defense_Strength': [], 'H2H_HomeWins_Pct': [], 'H2H_Draws_Pct': [], 'H2H_AwayWins_Pct': [], 'H2H_Home_AvgGoals': [], 'H2H_Away_AvgGoals': [] }

        # This iterative approach is correct but slow.
        for index, row in df.iterrows():
            home_team, away_team, match_date, season = row['HomeTeam'], row['AwayTeam'], row['MatchDate'], row['Season']
            past_df = df[df.index < index]
            
            home_matches = past_df[((past_df['HomeTeam'] == home_team) | (past_df['AwayTeam'] == home_team))].tail(5)
            away_matches = past_df[((past_df['HomeTeam'] == away_team) | (past_df['AwayTeam'] == away_team))].tail(5)
            
            # Form
            home_form_points = home_matches.apply(lambda r: r['HomePoints'] if r['HomeTeam'] == home_team else r['AwayPoints'], axis=1)
            feature_lists['HomeTeam_Form'].append(home_form_points.sum())
            away_form_points = away_matches.apply(lambda r: r['HomePoints'] if r['HomeTeam'] == away_team else r['AwayPoints'], axis=1)
            feature_lists['AwayTeam_Form'].append(away_form_points.sum())
            
            # Weighted Form
            hwf, awf = 0, 0
            prev_s_str = f"{int(season.split('/')[0]) - 1}/{str(int(season.split('/')[1]) - 1).zfill(2)}"
            max_r = len(season_ranks.get(prev_s_str, {}))
            for i, pr in home_matches.iterrows():
                opp = pr['AwayTeam'] if pr['HomeTeam'] == home_team else pr['HomeTeam']
                opp_r = season_ranks.get(prev_s_str, {}).get(opp, max_r + 1)
                w = (max_r - opp_r + 1) / max_r if max_r > 0 else 1
                hwf += home_form_points.loc[i] * w if i in home_form_points.index else 0
            feature_lists['Home_Weighted_Form'].append(hwf)
            for i, pr in away_matches.iterrows():
                opp = pr['AwayTeam'] if pr['HomeTeam'] == away_team else pr['HomeTeam']
                opp_r = season_ranks.get(prev_s_str, {}).get(opp, max_r + 1)
                w = (max_r - opp_r + 1) / max_r if max_r > 0 else 1
                awf += away_form_points.loc[i] * w if i in away_form_points.index else 0
            feature_lists['Away_Weighted_Form'].append(awf)

            # Strength
            la = league_averages.get(season, {'avg_home_goals': 1, 'avg_away_goals': 1})
            hs = home_matches.apply(lambda r: r['FullTimeHomeGoals'] if r['HomeTeam'] == home_team else r['FullTimeAwayGoals'], axis=1).mean()
            hc = home_matches.apply(lambda r: r['FullTimeAwayGoals'] if r['HomeTeam'] == home_team else r['FullTimeHomeGoals'], axis=1).mean()
            as_ = away_matches.apply(lambda r: r['FullTimeHomeGoals'] if r['HomeTeam'] == away_team else r['FullTimeAwayGoals'], axis=1).mean()
            ac = away_matches.apply(lambda r: r['FullTimeAwayGoals'] if r['HomeTeam'] == away_team else r['FullTimeHomeGoals'], axis=1).mean()
            feature_lists['Home_Attack_Strength'].append(hs / la['avg_home_goals'] if la['avg_home_goals'] > 0 and not np.isnan(hs) else 1)
            feature_lists['Home_Defense_Strength'].append(hc / la['avg_away_goals'] if la['avg_away_goals'] > 0 and not np.isnan(hc) else 1)
            feature_lists['Away_Attack_Strength'].append(as_ / la['avg_away_goals'] if la['avg_away_goals'] > 0 and not np.isnan(as_) else 1)
            feature_lists['Away_Defense_Strength'].append(ac / la['avg_home_goals'] if la['avg_home_goals'] > 0 and not np.isnan(ac) else 1)

            # H2H
            h2h = past_df[(((past_df['HomeTeam'] == home_team) & (past_df['AwayTeam'] == away_team)) | ((past_df['HomeTeam'] == away_team) & (past_df['AwayTeam'] == home_team)))]
            if not h2h.empty:
                h_w = sum((h2h['HomeTeam'] == home_team) & (h2h['FullTimeResult'] == 'H')) + sum((h2h['AwayTeam'] == home_team) & (h2h['FullTimeResult'] == 'A'))
                draws = sum(h2h['FullTimeResult'] == 'D')
                a_w = len(h2h) - h_w - draws
                feature_lists['H2H_HomeWins_Pct'].append(h_w / len(h2h)); feature_lists['H2H_Draws_Pct'].append(draws / len(h2h)); feature_lists['H2H_AwayWins_Pct'].append(a_w / len(h2h))
                feature_lists['H2H_Home_AvgGoals'].append(h2h.apply(lambda r: r['FullTimeHomeGoals'] if r['HomeTeam'] == home_team else r['FullTimeAwayGoals'], axis=1).mean())
                feature_lists['H2H_Away_AvgGoals'].append(h2h.apply(lambda r: r['FullTimeAwayGoals'] if r['HomeTeam'] == home_team else r['FullTimeHomeGoals'], axis=1).mean())
            else:
                for k in ['H2H_HomeWins_Pct', 'H2H_Draws_Pct', 'H2H_AwayWins_Pct', 'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals']: feature_lists[k].append(0)
        return feature_lists

    def get_dataloaders(self, batch_size=32):
        if self.processed_data is None: return None, None
        feature_cols = ['HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards', 'HomeTeam_Form', 'AwayTeam_Form', 'Match_of_the_Season', 'H2H_HomeWins_Pct', 'H2H_Draws_Pct', 'H2H_AwayWins_Pct', 'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals', 'Home_Attack_Strength', 'Home_Defense_Strength', 'Away_Attack_Strength', 'Away_Defense_Strength', 'Home_Weighted_Form', 'Away_Weighted_Form']
        
        for col in feature_cols:
            if col not in self.processed_data.columns: self.processed_data[col] = 0
            if pd.api.types.is_numeric_dtype(self.processed_data[col]): self.processed_data[col].fillna(0, inplace=True)
        
        self.processed_data.dropna(subset=['FTR_numerical', 'HomeTeam_ID', 'AwayTeam_ID'], inplace=True)

        seasons = sorted(self.processed_data['Season'].unique())
        latest_season = seasons[-1]
        train_df, test_df = self.processed_data[self.processed_data['Season'] != latest_season], self.processed_data[self.processed_data['Season'] == latest_season]
        
        X_train, y_train, home_ids_train, away_ids_train = train_df[feature_cols].values, train_df['FTR_numerical'].values, train_df['HomeTeam_ID'].values.astype(int), train_df['AwayTeam_ID'].values.astype(int)
        X_test, y_test, home_ids_test, away_ids_test = test_df[feature_cols].values, test_df['FTR_numerical'].values, test_df['HomeTeam_ID'].values.astype(int), test_df['AwayTeam_ID'].values.astype(int)

        train_dataset = PremierLeagueDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), torch.tensor(home_ids_train, dtype=torch.long), torch.tensor(away_ids_train, dtype=torch.long))
        test_dataset = PremierLeagueDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long), torch.tensor(home_ids_test, dtype=torch.long), torch.tensor(away_ids_test, dtype=torch.long))
        
        train_loader, test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"Training on seasons before {latest_season}. Validating on season {latest_season}.")
        return train_loader, test_loader