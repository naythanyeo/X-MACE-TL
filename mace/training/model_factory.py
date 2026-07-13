"""
Model factory helper function to initialise model from default parameters
Use two types of presets, either the default from x-mace CLI, or a lightweight
version that can be used for testing. 
Alternative, the presets can be mmanually overwritten when it is called. 

AtomDataMetadata is expected when initialising the model. 
Get the metadata object from builder.metadata
"""

from typing import List, Optional, Union

from e3nn import o3

from mace import modules
from mace.data.atom_data_loader import AtomDataMetadata

"""
Pre-defined model presets that match the X-Mace CLI
"""
_AUTOENCODER_PRESETS = {
    "default": {
        "latent_dim": 16,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 3,
        "num_interactions": 2,
        "hidden_irreps": "128x0e + 128x1o",
        "mlp_irreps": "16x0e",
        "correlation": 3,
        "gate": "silu",
        "interaction": "RealAgnosticResidualInteractionBlock",
        "interaction_first": "RealAgnosticResidualInteractionBlock",
        "radial_mlp": [64, 64, 64],
        "radial_type": "bessel",
        "distance_transform": "None",
        "pair_repulsion": False,
    },
    "lightweight": {
        "latent_dim": 16,
        "num_bessel": 4,
        "num_polynomial_cutoff": 3,
        "max_ell": 2,
        "num_interactions": 1,
        "hidden_irreps": "4x0e + 4x1o",
        "mlp_irreps": "16x0e",
        "correlation": 1,
        "gate": "silu",
        "interaction": "RealAgnosticResidualInteractionBlock",
        "interaction_first": "RealAgnosticResidualInteractionBlock",
        "radial_mlp": [32, 32],
        "radial_type": "bessel",
        "distance_transform": "None",
        "pair_repulsion": False,
    },
}


def initialise_autoencoder(
    metadata: AtomDataMetadata,
    preset: str = "default",
    latent_dim: Optional[int] = None,
    num_bessel: Optional[int] = None,
    num_polynomial_cutoff: Optional[int] = None,
    max_ell: Optional[int] = None,
    num_interactions: Optional[int] = None,
    hidden_irreps: Optional[Union[str, o3.Irreps]] = None,
    mlp_irreps: Optional[Union[str, o3.Irreps]] = None,
    correlation: Optional[Union[int, List[int]]] = None,
    gate: Optional[str] = None,
    interaction: Optional[str] = None,
    interaction_first: Optional[str] = None,
    radial_mlp: Optional[List[int]] = None,
    radial_type: Optional[str] = None,
    distance_transform: Optional[str] = None,
    pair_repulsion: Optional[bool] = None,
) -> modules.AutoencoderExcitedMACE:
    # Verify that preset selected is available 
    # Currently only default or lightweight
    if preset not in _AUTOENCODER_PRESETS:
        valid_presets = ", ".join(_AUTOENCODER_PRESETS)
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {valid_presets}.")

    # Use the default preset settings
    settings = _AUTOENCODER_PRESETS[preset].copy()
    overrides = {
        "latent_dim": latent_dim,
        "num_bessel": num_bessel,
        "num_polynomial_cutoff": num_polynomial_cutoff,
        "max_ell": max_ell,
        "num_interactions": num_interactions,
        "hidden_irreps": hidden_irreps,
        "mlp_irreps": mlp_irreps,
        "correlation": correlation,
        "gate": gate,
        "interaction": interaction,
        "interaction_first": interaction_first,
        "radial_mlp": radial_mlp,
        "radial_type": radial_type,
        "distance_transform": distance_transform,
        "pair_repulsion": pair_repulsion,
    }

    # Override any parameters that were specified by the factory input
    settings.update(
        {name: value for name, value in overrides.items() if value is not None}
    )

    # Apart from preset settings, also use the info from metadata to define model
    return modules.AutoencoderExcitedMACE(
        r_max=metadata.r_max,
        num_bessel=settings["num_bessel"],
        num_polynomial_cutoff=settings["num_polynomial_cutoff"],
        num_permutational_invariant=settings["latent_dim"],
        n_energies=metadata.n_energies,
        compute_nacs=False,
        compute_socs=False,
        soc_num=0,
        nac_num=0,
        max_ell=settings["max_ell"],
        interaction_cls=modules.interaction_classes[settings["interaction"]],
        interaction_cls_first=modules.interaction_classes[
            settings["interaction_first"]
        ],
        num_interactions=settings["num_interactions"],
        num_elements=metadata.num_elements,
        hidden_irreps=o3.Irreps(settings["hidden_irreps"]),
        MLP_irreps=o3.Irreps(settings["mlp_irreps"]),
        atomic_energies=metadata.atomic_energies,
        avg_num_neighbors=metadata.avg_num_neighbors,
        atomic_numbers=metadata.atomic_numbers,
        correlation=settings["correlation"],
        gate=modules.gate_dict[settings["gate"]],
        pair_repulsion=settings["pair_repulsion"],
        distance_transform=settings["distance_transform"],
        radial_MLP=settings["radial_mlp"],
        radial_type=settings["radial_type"],
    )
