"""
Model factory helper function to initialise model from default parameters
Currently has 3 different presets, either the default from x-mace CLI, 
the default that the ANI model uses, or a lightweight version suitable for
testing on local devices

Alternative, the presets can be mmanually overwritten when it is called. 

AtomDataMetadata is expected when initialising the model. 
Get the metadata object from builder.metadata
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
from e3nn import o3

from mace import modules
from mace.data.atom_data_loader import AtomDataMetadata
from mace.tools.finetuning_utils import load_foundations

from .model_presets import AUTOENCODER_PRESETS


FOUNDATION_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "calculators" / "foundations_models"
)
FOUNDATION_MODELS = {
    "ani500k": FOUNDATION_MODEL_DIR / "ani500k_large_CC_cpu.model",
    "mace_mp": FOUNDATION_MODEL_DIR / "2023-12-03-mace-mp.model",
}


def _load_base_model(load_base: str) -> torch.nn.Module:
    """
    Load one of the bundled base models.
    load_base is either a keyword ani500k/mace_mp or a file path 
    """
    if load_base in FOUNDATION_MODELS.keys():
        model_path = FOUNDATION_MODELS[load_base]
    else:
        model_path = Path(load_base)
    if not model_path.is_file():
        raise FileNotFoundError(f"Foundation model not found: {model_path}.")

    model = torch.load(model_path, map_location="cpu", weights_only=False)

    return model


def _backbone_parameters(model: torch.nn.Module) -> dict:
    """
    Extract foundation parameters using the model factory argument names.
    """
    radial_types = {
        "BesselBasis": "bessel",
        "GaussianBasis": "gaussian",
        "ChebychevBasis": "chebyshev",
    }
    distance_transforms = {
        "AgnesiTransform": "Agnesi",
        "SoftTransform": "Soft",
    }
    interactions = list(model.interactions)
    products = list(model.products)
    radial_name = model.radial_embedding.bessel_fn.__class__.__name__
    distance_name = (
        model.radial_embedding.distance_transform.__class__.__name__
        if hasattr(model.radial_embedding, "distance_transform")
        else "None"
    )
    correlations = [
        len(product.symmetric_contractions.contractions[0].weights) + 1
        for product in products
    ]
    radial_mlps = [list(block.conv_tp_weights.hs[1:-1]) for block in interactions]

    return {
        "r_max": float(model.r_max.item()),
        "radial_type": radial_types.get(radial_name, radial_name),
        "num_bessel": int(model.radial_embedding.out_dim),
        "num_polynomial_cutoff": int(model.radial_embedding.cutoff_fn.p.item()),
        "distance_transform": distance_transforms.get(distance_name, distance_name),
        "max_ell": int(model.spherical_harmonics._lmax),
        "hidden_irreps": str(products[0].linear.irreps_out),
        "num_interactions": len(interactions),
        "interaction": interactions[-1].__class__.__name__,
        "interaction_first": interactions[0].__class__.__name__,
        "correlation": correlations[0] if len(set(correlations)) == 1 else correlations,
        "radial_mlp": (
            radial_mlps[0]
            if all(value == radial_mlps[0] for value in radial_mlps)
            else radial_mlps
        ),
    }


def _validate_parameters(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    z_table,
) -> None:
    """Check that the target and base-model backbones are compatible."""
    model_parameters = _backbone_parameters(model)
    foundation_parameters = _backbone_parameters(base_model)
    mismatches = []

    # Check the architecture parameters
    for (key, model_value), (_, foundation_value) in zip(
        model_parameters.items(), foundation_parameters.items()
    ):
        if model_value != foundation_value:
            mismatches.append(
                f"{key}={foundation_value!r} is required; got {model_value!r}."
            )

    # Check for the atoms support
    foundation_elements = {int(z) for z in base_model.atomic_numbers}
    unsupported_elements = [
        int(z) for z in z_table.zs if int(z) not in foundation_elements
    ]
    if unsupported_elements:
        mismatches.append(
            f"atomic_numbers must be a subset of {sorted(foundation_elements)}; "
            f"got unsupported elements {unsupported_elements}."
        )

    if mismatches:
        raise ValueError("Base model parameters do not match:\n" + "\n".join(mismatches))


def initialise_autoencoder(
    metadata: AtomDataMetadata,
    preset: str = "default_ani",
    compute_nacs: bool = False,
    nac_num: int = 0,
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
    load_base: Optional[str] = None,
) -> modules.AutoencoderExcitedMACE:
    # Verify that preset selected is available 
    # Currently only default or lightweight
    if preset not in AUTOENCODER_PRESETS:
        valid_presets = ", ".join(AUTOENCODER_PRESETS)
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {valid_presets}.")

    # Use the default preset settings
    settings = AUTOENCODER_PRESETS[preset].copy()
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
    model = modules.AutoencoderExcitedMACE(
        r_max=metadata.r_max,
        num_bessel=settings["num_bessel"],
        num_polynomial_cutoff=settings["num_polynomial_cutoff"],
        num_permutational_invariant=settings["latent_dim"],
        n_energies=metadata.n_energies,
        compute_nacs=compute_nacs,
        compute_socs=False,
        soc_num=0,
        nac_num=nac_num,
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

    if load_base is not None:
        base_model = _load_base_model(load_base)
        _validate_parameters(
            model=model,
            base_model=base_model,
            z_table=metadata.z_table,
        )

        # Shift model dtype to match dtype of foundation model first
        # Higher precision when loading in parameters
        # But save the original models device and dtype to convert back
        og_model_paramter = next(model.parameters())
        og_device = og_model_paramter.device
        og_dtype = og_model_paramter.dtype
        
        base_model_parameter = next(base_model.parameters())
        model = model.to(
            device=base_model_parameter.device,
            dtype=base_model_parameter.dtype,
        )
        max_L = o3.Irreps(settings["hidden_irreps"]).lmax

        model = load_foundations(
            model=model,
            model_foundations=base_model,
            table=metadata.z_table,
            load_readout=False,
            use_shift=False,
            use_scale=True,
            max_L=max_L,
        )
        model = model.to(
            device=og_device,
            dtype=og_dtype
        )

    return model
