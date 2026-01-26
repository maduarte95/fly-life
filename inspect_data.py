import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


trajectories_path_1 = 'data/trajectories/0_trajectories/Copy of DGRP375_CamAVideo1_2026-01-12T08_31_36_trajectories.csv'
trajectories_path_2 = 'data/trajectories/0_trajectories/Copy of DGRP375_CamAVideo2_2026-01-12T08_31_36_trajectories.csv'
areas_path = 'data/trajectories/0_trajectories/Copy of DGRP375_CamAVideo1_2026-01-12T08_31_36_areas.csv'

areas = load_data(areas_path)
trajectories_1 = load_data(trajectories_path_1)
trajectories_2 = load_data(trajectories_path_2)
trajectories = pd.concat([trajectories_1, trajectories_2], ignore_index=True)

# %%
#display
display(areas.head())
print(areas.info())
#print length 
print(f"Length of areas DataFrame: {len(areas)}")
print("Columns in areas DataFrame:", areas.columns.tolist() )
# identify row with the smallest "mean"
male = areas.loc[areas['mean'].idxmin()]
print("Row with the smallest 'mean':", male)
# actual male id is the previous one plus one
male_id = male.name + 1
print("Index of the male: ", male_id)
# %%
display(trajectories.head())
print(trajectories.info())
print(f"Length of trajectories DataFrame: {len(trajectories)}")
print("Columns in trajectories DataFrame:", trajectories.columns.tolist() )

#%%
recording_time = 7200 # seconds, 2 hours
trajectory_length = len(trajectories)
frame_rate = trajectory_length / recording_time
print(f"Frame rate: {frame_rate} frames per second")
# %%
