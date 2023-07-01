# Python file to generate training examples for the brain fmri to caption problem

import os
import sys
import numpy as np
import pickle as pkl


def load_fmri_data(path: str, brain_locations: list =['WB']) -> dict:
    """Loads fMRI pickles. Returns dict with all loaded pickles with brain locations as keys."""

    fmri_data = {}

    for p in os.listdir(path):
        if '56789' not in p: continue
        
        brain_region = p.split('_')[0]

        if brain_region in brain_locations:
            with open(os.path.join(path, p), 'rb') as f:
                data = pkl.load(f)
                fmri_data[brain_region] = data

    return fmri_data
        

def main():

    # Load the fmri data to use
    


    # Concatenate into one large object

    # Save pickle

    
if __name__ == "__main__":
    #parse arguments



    main()