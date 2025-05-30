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