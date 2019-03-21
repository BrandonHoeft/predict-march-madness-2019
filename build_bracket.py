from bracketeer import build_bracket
import os
from os.path import expanduser

b = build_bracket(
        outputPath='output.png',
        teamsPath='data/Teams.csv',
        seedsPath='data/TourneySeeds.csv',
        submissionPath='data/submit.csv',
        slotsPath='data/TourneySlots.csv',
        year=2017
)

# Running Below assumes that repo is stored on Macbook Desktop directory.
home = expanduser("~") # root on my machine
src_dir = home + '/Desktop/predict-march-madness-2019/stage2_data/'
img_dir = home + '/Desktop/predict-march-madness-2019/images/'

# Bracket Based upon the XGBoosted Model predictions
XGB_bracket = build_bracket(outputPath=img_dir+'xgboost_bracket.png',
                            teamsPath=src_dir+'Teams.csv',
                            seedsPath=src_dir+'NCAATourneySeeds.csv',
                            submissionPath=src_dir+'stage2_XGBoost_submission.csv',
                            slotsPath=src_dir+'NCAATourneySlots.csv',
                            year=2019)



# Bracket Based upon the Stacked Model predictions (Penalized Logistic + RF base learners)
STACKED_model_bracket = build_bracket(outputPath=img_dir+'stacked_model_bracket.png',
                                      teamsPath=src_dir+'Teams.csv',
                                      seedsPath=src_dir+'NCAATourneySeeds.csv',
                                      submissionPath=src_dir+'stage2_STACKED_submission.csv',
                                      slotsPath=src_dir+'NCAATourneySlots.csv',
                                      year=2019)

