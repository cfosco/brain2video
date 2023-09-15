# Brain2video project
Repo for the Brain2Video reconstruction project

The main objective is to reconstruct videos as accurately as possible from fMRI data of participants watching the video. We use the BOLDMoments dataset for our main experiments.

Our approach essentially attempts to regress conditioning and latent embeddings from fMRI that can then be fed to a pretrained Text2Video or Video2Video generation model to obtain a representation of what the user was seeing.

## Download data
Use `download_data.py` to download the BOLDMoments data if you have an MIT kerberos account.

## Extract targets
The first step is to extract conditioning and latent embeddings, which we call targets (targets for the regression problem). This can be done with the `extract_*.py` scripts. The scripts will save the targets under`data/target_vectors`.

## Fit regression model to go from fMRI into target vectors
With the computed embeddings, we have our ground truth for the regression problem. We then attempt to reconstruct these ground truth vectors from the fMRI data. We use different types of fMRI data, such as betas and blood flow. To train your regressor, use `regress.py`. 

## Reconstruct video
The trained regressor allows us to generate conditioning and latent embeddings for a new test-set fMRI. These embeddings are then fed to a generation model to reconstruct the video. Use `reconstruct_zeroscope.py` to generate reconstructions with the zeroscope_v2_576w model, for example.

