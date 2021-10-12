# -*- coding: utf-8 -*-
"""
MVPA analysis of NEAR-project - adult sample. Decoding small vs. big non-tie problems.
========================================== 
Specificities of the decoding pipeline: 
    - use beta values from first-level analysis
    - use searchlight method for detecting interesting brain regions
    - Use K-fold training procedure to train and test our model. 

@author: Katrien Vandenbroeck
"""
from nilearn import plotting, decoding
from nilearn.plotting import view_img
from nilearn.image import new_img_like
import nilearn
import numpy as np 
import os
import pandas as pd
from sklearn import model_selection

#%% select beta-maps from first-level analysis
## define path to file
file   = "S:\\bbl\\NTR\\NEAR_BIDS"
## & files corresponding to adult sample
adult_nb = ['sub-0'+str(i) for i in np.arange(0,10)]+['sub-'+str(i) for i in np.arange(10,46)]
adults   = [file+'\\'+f+'\\analysis2' for f in os.listdir(file) if f in (nb for nb in adult_nb)]

##define beta-maps (main effect of small/big non-tie)
# (beta_0001: first run, main effect small non tie; 
# beta_0002: first run, main effect big non tie etc.)
main_small_big = ['beta_0001.nii','beta_0003.nii','beta_0009.nii','beta_0011.nii',
                  'beta_0017.nii','beta_0019.nii','beta_0025.nii','beta_0027.nii',
                  'beta_0033.nii','beta_0035.nii'] 
#####
#! note: not all participants did all 5 runs. 
# so we extract the beta value depending on the number of runs the participant did. 
#####

## select relevant files: (i.e. small and big non tie main effect)
betas_file = []
conditions = []
run        = []
for subject in adults[0:5]:
    all_betas = [b for b in os.listdir(subject) if 'beta' in b]
    if all_betas[-1] == 'beta_0045.nii': ## check if last beta-map is equal to 45; indicates 5 runs.
        betas_file += [subject+'\\'+b for b in all_betas if b in (f for f in main_small_big)]
        conditions += ['small_nontie','big_nontie']*5
        run += list(np.repeat(np.arange(1,6),2))
    elif all_betas[-1] == 'beta_0036.nii': ## indicates 4 runs
        betas_file += [subject+'\\'+b for b in all_betas if b in (f for f in main_small_big[0:8])]
        conditions += ['small_nontie','big_nontie']*4
        run += list(np.repeat(np.arange(1,5),2))
data = pd.DataFrame(data={'fmri':betas_file,'label':conditions,'run':run})

#%% create a mask to extract the brain from non-brain info
mask = nilearn.masking.compute_multi_background_mask(betas_file)
## check fit for mask
plotting.plot_roi(mask,betas_file[4])

#%% do searchlight analysis & decoding
# we will use a leave-one-run-out cross-validation method.
## define training and test data
cross_validation = model_selection.LeaveOneGroupOut()
searchlight      = decoding.SearchLight(mask,radius = 5.6, n_jobs = 1, verbose = 1, cv = cross_validation)
searchlight.fit(data['fmri'],data['label'], groups=data['run'])

#%% visualizing the results:
results = new_img_like(mask, searchlight.scores_)
view_img(results,threshold=0.8).open_in_browser()

