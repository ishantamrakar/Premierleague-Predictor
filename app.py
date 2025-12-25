"""
Premier League Match Predictor - Simple Streamlit App
Author: Your Name
Description: Select two teams and get AI-powered match predictions!
"""

import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
import sys
import importlib
from pathlib import Path

# Import configuration
import config

# Dynamically import the model architecture from the active model
model_module = importlib.import_module(config.ARCHITECTURE_MODULE)
MatchPredictor = model_module.MatchPredictor

# Page configuration
st.set_page_config(
    page_title="Premier League Predictor",
    layout="centered",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #38003c;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prob-bar {
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Premier League Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered match outcome predictions using neural networks</div>', unsafe_allow_html=True)

# Load model and data (cached for performance)
@st.cache_resource
def load_model_and_data():
    """Load the trained model, team mappings, and pre-computed features"""
    # Get model info
    model_info = config.get_model_info()

    # Load team mappings
    with open(config.TEAM_MAPPING_PATH, 'r') as f:
        team_to_id = json.load(f)

    # Load pre-computed features
    features_df = pd.read_csv(config.FEATURES_PATH)

    # Load model
    state_dict = torch.load(config.MODEL_PATH)
    EMBEDDING_DIM = model_info.get('embedding_dim', 10)
    input_dim = state_dict['fc1.weight'].shape[1] - (2 * EMBEDDING_DIM)
    num_teams = len(team_to_id)

    model = MatchPredictor(input_dim=input_dim, num_teams=num_teams, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(state_dict)
    model.eval()

    return model, team_to_id, features_df, model_info

# Load everything
with st.spinner("Loading AI model... Please wait..."):
    try:
        model, team_to_id, features_df, model_info = load_model_and_data()
        st.success(f"Model loaded: {model_info.get('name', 'Unknown')} v{model_info.get('version', '1.0')}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Get list of valid teams
valid_teams = sorted(list(team_to_id.keys()))

# Team selection UI
st.markdown("---")
st.markdown("### Select Match")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "Home Team",
        valid_teams,
        index=valid_teams.index("Man United") if "Man United" in valid_teams else 0
    )

with col2:
    away_team = st.selectbox(
        "Away Team",
        valid_teams,
        index=valid_teams.index("Arsenal") if "Arsenal" in valid_teams else 1
    )

# Predict button
if st.button("Predict Match Outcome", type="primary", use_container_width=True):
    # Validate teams are different
    if home_team == away_team:
        st.error("Please select two different teams!")
    else:
        with st.spinner("Running prediction..."):
            # Look up features for this match
            match_row = features_df[
                (features_df['HomeTeam'] == home_team) &
                (features_df['AwayTeam'] == away_team)
            ]

            if match_row.empty:
                st.warning(f"No pre-computed features found for {home_team} vs {away_team}")
                st.info("This match may not be in the upcoming fixtures list.")
            else:
                # Extract features
                match_features = match_row.iloc[0]
                feature_cols = [
                    'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
                    'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
                    'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
                    'HomeTeam_Form', 'AwayTeam_Form', 'Match_of_the_Season',
                    'H2H_HomeWins_Pct', 'H2H_Draws_Pct', 'H2H_AwayWins_Pct',
                    'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals', 'Home_Attack_Strength',
                    'Home_Defense_Strength', 'Away_Attack_Strength', 'Away_Defense_Strength',
                    'Home_Weighted_Form', 'Away_Weighted_Form'
                ]

                features_tensor = torch.tensor(
                    match_features[feature_cols].values.astype(np.float32),
                    dtype=torch.float32
                )

                # Get team IDs
                home_id = team_to_id[home_team]
                away_id = team_to_id[away_team]
                home_id_tensor = torch.tensor([home_id], dtype=torch.long)
                away_id_tensor = torch.tensor([away_id], dtype=torch.long)

                # Run prediction
                with torch.no_grad():
                    output = model(features_tensor.unsqueeze(0), home_id_tensor, away_id_tensor)
                    probs = torch.softmax(output, dim=1).squeeze()
                    prediction_idx = torch.argmax(probs).item()

                    # Map indices to outcomes
                    idx_to_outcome = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
                    predicted_outcome = idx_to_outcome[prediction_idx]

                    # Get probabilities
                    prob_home = probs[2].item()
                    prob_draw = probs[1].item()
                    prob_away = probs[0].item()

                # Display results
                st.markdown("---")
                st.markdown("### Prediction Results")

                # Main prediction
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2>{predicted_outcome}</h2>'
                    f'<p style="font-size: 1.2rem; margin-top: 1rem;">'
                    f'{home_team} vs {away_team}'
                    f'</p></div>',
                    unsafe_allow_html=True
                )

                # Probability breakdown
                st.markdown("#### Probability Breakdown")

                # Home Win
                st.markdown(f"**{home_team} Win**")
                st.progress(prob_home)
                st.caption(f"{prob_home:.1%}")

                # Draw
                st.markdown(f"**Draw**")
                st.progress(prob_draw)
                st.caption(f"{prob_draw:.1%}")

                # Away Win
                st.markdown(f"**{away_team} Win**")
                st.progress(prob_away)
                st.caption(f"{prob_away:.1%}")

                # Model confidence
                confidence = max(prob_home, prob_draw, prob_away)
                st.markdown("---")
                st.metric("Model Confidence", f"{confidence:.1%}")

                # Feature insights (expandable)
                with st.expander("View Feature Insights"):
                    st.markdown("**Team Form (Last 5 matches)**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{home_team} Form", f"{match_features['HomeTeam_Form']:.1f} pts")
                    with col2:
                        st.metric(f"{away_team} Form", f"{match_features['AwayTeam_Form']:.1f} pts")

                    st.markdown("**Head-to-Head Record**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Home Wins", f"{match_features['H2H_HomeWins_Pct']:.0%}")
                    with col2:
                        st.metric("Draws", f"{match_features['H2H_Draws_Pct']:.0%}")
                    with col3:
                        st.metric("Away Wins", f"{match_features['H2H_AwayWins_Pct']:.0%}")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p><strong>Model:</strong> {model_info.get('name', 'Unknown')} v{model_info.get('version', '1.0')} |
        <strong>Accuracy:</strong> {model_info.get('accuracy', 'N/A')}</p>
        <p>{model_info.get('description', 'AI-powered predictions')}</p>
        <p><strong>Disclaimer:</strong> Predictions are for entertainment purposes only</p>
    </div>
""", unsafe_allow_html=True)
