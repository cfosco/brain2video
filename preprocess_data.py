# Python file to generate training examples for the brain fmri to caption problem

import os
import sys
import numpy as np
import pickle as pkl
import argparse


def load_fmri_data(path: str, brain_regions_to_load: list = None) -> dict:
    """Loads fMRI pickles. Returns dict with all loaded pickles with brain regions as keys."""

    fmri_data = {}

    for p in os.listdir(path):
        if '56789' not in p: continue
        
        brain_region = p.split('_')[0]

        if brain_regions_to_load is not None and brain_region in brain_regions_to_load:
            with open(os.path.join(path, p), 'rb') as f:
                data = pkl.load(f)
                fmri_data[brain_region] = data

    return fmri_data
        

def main(args):

    # Load the fmri data to use
    fmri_data = load_fmri_data(path=args.path, brain_region=args.areas_to_concatenate)

    # Concatenate into one large object
    br_concat = np.concatenate([fmri_data[brain_region] for brain_region in fmri_data.keys()], axis=2)

    # Save pickle
    save_name = '_'.join(fmri_data.keys())
    with open(os.path.join(args.save_path, f'{save_name}.pkl'), 'wb') as f:
        pkl.dump(br_concat, f)
    

    
if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Concatenate brain regions and save as pickle for training')
    parser.add_argument('--path', type=str, help='Path to the brain region pickles')
    parser.add_argument('--path', type=str, help='Path to save the concatenated pickle')
    parser.add_argument('--regions', type=str, help='Brain regions to use. Enter brain regions separated by commas, e.g. "V1a,V1b,PPA"')
    args = parser.parse_args()

    main(args)