# Fly courtship behavior

## Scripts
inspect_data.py: example for loading data
ethogram.py: create an ethogram, identify segments

Run with uv run

## Data
- Data is in the data/ directory.
- Videos are in data/videos/
- Tracked videos are in data/tracked_videos/
- Trajectories are in data/trajectories/0_trajectories

**Structure of experimental runs**
- 1 run: 2 videos; 4 csv files;
- Experimental conditions: CamA and CamB
- Video names: 
- Trajectory names:  DGRP375_Cam{CamId}Video{1 or 2}_date_{areas/trajectories}.csv
    - E.g. DGRP375_CamAVideo1_2026-01-12T08_31_36_areas.csv


## Analyses
- Time to copulation
    - In general (any male to any copulation)
    - For each female
- Pursuit
    - % time pursuing each female
    - % time pursuing in general

Plots: ethograms where x axis is time, y axis is female fly id???

# GOAL 1
- Make an ethogram for the first run, camA!
- example code in inspect_data.py

- get distances from male to each female
- define thresholds for copulation detection
- define thresholds for pursuit
- plot ethogram
- inspect the resulting ethogram


## Notes for analysis
- In copulation:
    - we can have two (male and F) but very close (distance threshold) for >1min
    - OR we can have 5 ids instead of 6 (5 ids or less for >1min); check which female is being copulated:
        - The male can disappear or
        - Female can disappear (in which case the missing id is te female being copulated)
        - Other ids can also disappear (but probably less than 1min)

- For pursuit:
    - Distance threshold
    - Period of time threshold (e.g. 2s)

- Convert to cm:
    - Diameter around 850 pixels; 425 radius (with paint, maybe incorrect)
    - 6.7 diameter arena
    - Slides:
        - CamA: 1cm = 208px
        - CamB: 1cm = 203px