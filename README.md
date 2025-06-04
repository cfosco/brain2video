# Brain2video project
Repo for the Brain2Video reconstruction project

The main objective is to reconstruct videos as accurately as possible from fMRI data of participants watching the video. We use the BOLDMoments dataset for our main experiments.

Our approach essentially attempts to regress conditioning and latent embeddings from fMRI that can then be fed to a pretrained Text2Video or Video2Video generation model to obtain a representation of what the user was seeing. 

## Create environment
```
conda create -n BrainNetflix python=3.11
conda activate BrainNetflix
cd /BrainNetflix/project/folder
pip install -r requirements.txt 
```

## Download data
Use `download_data.py` to download the fMRI data and stimulus sets if you have an MIT kerberos account. Otherwise, follow these steps and save it in './data/'

1. Download preprocessed fMRI data for BMD [1], HAD [2], and CC2017 [3] datasets from this [Google Drive link](https://drive.google.com/drive/folders/1xeXDUoArZ0kq8JWr2b5gaqBMP06mqH5f?usp=sharing)

2. Download stimulus set (and agree to each datasets' terms of use) through these links:
    - [BOLD Moments Dataset stimuli](https://github.com/blahner/BOLDMomentsDataset)
    - [Human Actions Dataset](https://openneuro.org/datasets/ds004488)
    - [CC2017](https://drive.google.com/drive/folders/12VEbW9AdujKlzxD6zGJJ3eoOq2cd9GJw?usp=sharing) (video clips are split into 2s chunks)

The fMRI data is organized in dictionaries that organize the data by vertex, stimulus repeat, and stimulus. "Group41" refers to a subset of 41 ROIs from the [MMP1.0 atlas](https://pmc.ncbi.nlm.nih.gov/articles/PMC4990127/). [hcp_utils](https://rmldj.github.io/hcp-utils/) is a helpful package when dealing with fMRI data in this space using this atlas. Use the notebook 'notebooks/betas_into_indiv_files.ipynb' to further preprocess the data into individual files for easier modeling.

## Extract targets
The first step is to extract conditioning and latent embeddings, which we call targets (targets for the regression problem). This can be done with the `extract_*.py` scripts. The scripts will save the targets under`./data/target_vectors_<dataset>`.

```
python extract_targets_blip.py --dataset bmd
python extract_targets_blip.py --dataset cc2017

python extract_targets_zeroscope.py --dataset bmd
python extract_targets_zeroscope.py --dataset cc2017
```

## Fit regression model to go from fMRI into target vectors
With the computed embeddings, we have our ground truth for the regression problem. We then attempt to reconstruct these ground truth vectors from the fMRI data. We use different types of fMRI data, such as betas and blood flow. To train your regressor, use `regress.py`.

## Reconstruct video
The trained regressor allows us to generate conditioning and latent embeddings for a new test-set fMRI. These embeddings are then fed to a generation model to reconstruct the video. Use `reconstruct_zeroscope.py` to generate reconstructions with the zeroscope_v2_576w model, for example.


## Pseudo code of the pipeline

```python
# Stage 1: MBM and Alignment
# Train the encoder-decoder to reconstruct masked fMRI embeddings
def train_mbm_encoder_decoder(fmri_data):
    encoder = MBMEncoder()
    decoder = MBMDecoder()

    # Masked Brain Modeling (MBM)
    masked_fmri_data = mask_patches(fmri_data)
    latent_vectors = encoder(masked_fmri_data)
    reconstructed_fmri = decoder(latent_vectors)
    mse_loss = calculate_mse_loss(reconstructed_fmri, fmri_data)

    optimize(encoder, decoder, mse_loss)
    return encoder

# Fine-tune the encoder with contrastive learning
def fine_tune_encoder_with_contrastive(encoder, fmri_data, clip_embeddings):
    latent_vectors = encoder(fmri_data)

    # Contrastive Learning
    contrastive_loss = calculate_contrastive_loss(latent_vectors, clip_embeddings)

    optimize(encoder, contrastive_loss)
    return encoder

# Stage 2: Regression
# Regress latent vector (z) and BLIP embeddings (b) from encoder outputs
def train_regressors(encoder_outputs):
    mlp_z = MLP()
    mlp_b = MLP()

    z = mlp_z(encoder_outputs)
    b = mlp_b(encoder_outputs)

    regression_loss_z = calculate_mse_loss(z, true_latent_vector)
    regression_loss_b = calculate_mse_loss(b, true_blip_embedding)

    optimize(mlp_z, mlp_b, regression_loss_z + regression_loss_b)
    return mlp_z, mlp_b

# Stage 3: Reconstruction
# Generate video using predicted latent vectors and BLIP embeddings
def reconstruct_video(new_fmri, encoder, mlp_z, mlp_b):
	fmri_embedding = encoder(new_fmri)
    latent_vector = mlp_z(frmri_embedding)
    blip_embedding = mlp_b(fmri_embedding)

    re_noised_z = renoise(latent_vector)
    caption = decode_blip(blip_embedding)

    video = denoise_video(re_noised_z, caption)
    return video

# Main pipeline function
def main_pipeline(fmri_data, clip_embeddings, new_fmri):
    # Stage 1: Train encoder-decoder with MBM
    encoder = train_mbm_encoder_decoder(fmri_data)

    # Stage 1: Fine-tune encoder with contrastive learning
    encoder = fine_tune_encoder_with_contrastive(encoder, fmri_data, clip_embeddings)

    # Stage 2: Train regressors for latent vector (z) and BLIP embeddings (b)
    encoder_outputs = encoder(fmri_data)
    mlp_z, mlp_b = train_regressors(encoder_outputs)

    # Stage 3: Reconstruct video from new fMRI data
    video = reconstruct_video(new_fmri, encoder, mlp_z, mlp_b)
    return video
```

### References

[1] Lahner, B., Dwivedi, K., Iamshchinina, P. et al. Modeling short visual events through the BOLD moments video fMRI dataset and metadata. Nat Commun 15, 6241 (2024). https://doi.org/10.1038/s41467-024-50310-3

[2] Zhou, M., Gong, Z., Dai, Y., Wen, Y., Liu, Y., & Zhen, Z. (2023). A large-scale fMRI dataset for human action recognition. Scientific Data, 10(1), 415.

[3] Wen, H., Shi, J., Zhang, Y., Lu, K. H., Cao, J., & Liu, Z. (2018). Neural encoding and decoding with deep learning for dynamic natural vision. Cerebral cortex, 28(12), 4136-4160.

### Citation

If you use this code, please cite our paper:

Fosco, C., Lahner, B., Pan, B., Andonian, A., Josephs, E., Lascelles, A., & Oliva, A. (2024, September). Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals. In European Conference on Computer Vision (pp. 457-474). Cham: Springer Nature Switzerland.