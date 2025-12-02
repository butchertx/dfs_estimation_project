import pandas as pd
import numpy as np


def get_projection_data():
    player_projections = []
    for week in range(1,19):
        data = pd.read_csv(f'data/projections_week_{week}.csv')
        player_projections.append(data)
    player_projections = pd.concat(player_projections)
    return player_projections

def get_dst_projection_data():
    dst_projections = []
    for week in range(1,19):
        data = pd.read_csv(f'data/dst_projections_week_{week}.csv')
        dst_projections.append(data)
    dst_projections = pd.concat(dst_projections)
    return dst_projections

def get_filtered_projection_data(include_team=False):
    player_projections = get_projection_data()
    dst_projections = get_dst_projection_data()
    player_rename_dict = {
        'Season': 'season',
        'Week': 'week',
        'Name': 'player_name',
        'Position': 'position',
        'FantasyPointsDraftKings_proj': 'projection'
    }
    team_rename_dict = {
        'Season': 'season',
        'Week': 'week',
        'Team': 'player_name',
        'FantasyPointsDraftKings_proj': 'projection'
    }
    if include_team:
        extra_cols = {
            'Team': 'team',
            'Opponent': 'opp_team',
            'HomeOrAway': 'home_or_away',
            'FantasyPointsDraftKings_act': 'points_scored'
        }
        player_rename_dict.update(extra_cols)
        team_rename_dict.update(extra_cols)

    player_projections = player_projections[player_rename_dict.keys()].rename(columns=player_rename_dict)
    dst_projections = dst_projections[team_rename_dict.keys()].rename(columns=team_rename_dict)
    if include_team:
        dst_projections['player_name'] = dst_projections['team']
    dst_projections['position'] = 'DST'
    return pd.concat([player_projections, dst_projections])

def get_model_data_for_regression(sample_size: int | None = None, random_sample=False):
    if sample_size is not None and not random_sample:
        data = pd.read_csv('data/project_data.csv', index_col=0 , nrows=sample_size)
    else:
        data = pd.read_csv('data/project_data.csv', index_col=0)
        if sample_size is not None and random_sample:
            data = data.sample(n=sample_size, random_state=42)
    filter_gt_0_usage = data['usage_ratio'] > 0.0
    usage_data_nonzero = data.loc[filter_gt_0_usage].copy()
    usage_data_nonzero['log_usage'] = np.log10(usage_data_nonzero['usage_ratio'])
    return usage_data_nonzero