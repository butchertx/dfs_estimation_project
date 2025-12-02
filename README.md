# Predicting Key Quantities for Daily Fantasy Football Contest Lineup Optimization using Machine Learning Models

Disclaimer: There's a lot of stuff in here that is not well-documented. Intended for reference only. 

- analysis.ipynb
    - Plots describing the distribution of contest stats\
- create_contest_usage_data.ipynb
    - Interactive notebook for compiling the source data into the combined format. Source data not provided
- create_project_data.ipynb
    - Used to create project_data.csv from source data
- data_wrangler.py
    - Helper methods for pulling in projection data from the weekly csv files
- evaluate_models.ipynb
    - Used for plotting the performance of the ML models 
- model_player_covariance.ipynb
    - plots describing the distributions of player projections and results
- models.py
    - helper code for handling the different models used for the usage prediction
- regression_training_benchmark.ipynb
    - used to estimate the training time for some ML models
- train_usage_nn.ipynb
    - used to train the neural networks for usage ratio
- usage_regression.ipynb
    - perform the model fitting for the non-NN models for usage ratio
 
## Data source

The data for this project can be found at the following link: [Google Drive](https://drive.google.com/drive/folders/1Q6RCjrUE4Zn1V0MlRjrrsNuI_4epd3db?usp=sharing)

There are .csv files containing projection data for offensive players and defensive teams for every week of the 2024 season. There is also a large csv file containing usage data for some Draftkings contests in the 2024 season.
