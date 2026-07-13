"""
Helper for extracting invariant latent representations.
This can be useful for sampling from latent space for eg? Rather than 
sampling from energies or geometric descriptors. 

After model is trained already, latent space can be reached as a tensor 
n_dim by num_geoms. 

Dataloader here important to be shuffle = False because currently we do not have
any labelling or special indices for the data yet
"""

import torch


def extract_latent_space(model, data_loader, device="cpu"):
    model.to(device)
    model.eval()
    latent_batches = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), training=False)
        latent_batches.append(output["invariant_vals"].detach().cpu())

    return torch.cat(latent_batches).numpy()
