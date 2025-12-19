import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os

class PremierLeagueDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class PremierLeagueDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self):
        """Loads raw Premier League match data."""
        try:
            csv_path = os.path.join(self.data_path, 'dataset/2000-2025 team stats/epl_final.csv')
            self.raw_data = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {csv_path}")
            self.raw_data = None

    def preprocess_data(self):
        """Preprocesses the raw data to calculate team form and prepare features."""
        if self.raw_data is None:
            return

        processed_df = self.raw_data.copy()

        processed_df['MatchDate'] = pd.to_datetime(processed_df['MatchDate'])
        processed_df.sort_values(by='MatchDate', inplace=True)

        result_mapping = {'A': 0, 'D': 1, 'H': 2}
        processed_df['FTR_numerical'] = processed_df['FullTimeResult'].map(result_mapping)
        
        home_points_map = {'H': 3, 'D': 1, 'A': 0}
        away_points_map = {'A': 3, 'D': 1, 'H': 0}
        processed_df['HomePoints'] = processed_df['FullTimeResult'].map(home_points_map)
        processed_df['AwayPoints'] = processed_df['FullTimeResult'].map(away_points_map)

        processed_df.dropna(subset=['FTR_numerical', 'HomePoints', 'AwayPoints'], inplace=True)

        home_form_list = []
        away_form_list = []

        for index, row in processed_df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['MatchDate']

            home_team_past_matches = processed_df[((processed_df['HomeTeam'] == home_team) | (processed_df['AwayTeam'] == home_team)) & (processed_df['MatchDate'] < match_date)].tail(5)
            home_team_points = home_team_past_matches.apply(lambda r: r['HomePoints'] if r['HomeTeam'] == home_team else r['AwayPoints'], axis=1)
            home_form_list.append(home_team_points.sum())

            away_team_past_matches = processed_df[((processed_df['HomeTeam'] == away_team) | (processed_df['AwayTeam'] == away_team)) & (processed_df['MatchDate'] < match_date)].tail(5)
            away_team_points = away_team_past_matches.apply(lambda r: r['HomePoints'] if r['HomeTeam'] == away_team else r['AwayPoints'], axis=1)
            away_form_list.append(away_team_points.sum())

        processed_df['HomeTeam_Form'] = home_form_list
        processed_df['AwayTeam_Form'] = away_form_list

        self.processed_data = processed_df

    def get_dataloaders(self, batch_size=32, test_size=0.2, random_state=42):
        """Returns train and test PyTorch DataLoaders."""
        if self.processed_data is None:
            return None, None

        feature_cols = [
            'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
            'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
            'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
            'HomeTeam_Form', 'AwayTeam_Form'
        ]
        
        existing_cols = [col for col in feature_cols if col in self.processed_data.columns]
        
        self.processed_data.dropna(subset=existing_cols + ['FTR_numerical'], inplace=True)

        X = self.processed_data[existing_cols].values
        y = self.processed_data['FTR_numerical'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = PremierLeagueDataset(X_train_tensor, y_train_tensor)
        test_dataset = PremierLeagueDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

if __name__ == '__main__':
    data_loader = PremierLeagueDataLoader(data_path='.')
    data_loader.load_data()
    if data_loader.raw_data is not None:
        data_loader.preprocess_data()
        if data_loader.processed_data is not None:
            print(data_loader.processed_data[[
                'MatchDate', 'HomeTeam', 'AwayTeam', 'HomeTeam_Form', 'AwayTeam_Form'
            ]].head())


