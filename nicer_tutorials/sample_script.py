"""
Updated sample script that does k fold validation training 
Mainly introduce all the new training paramters to match github as 
best as possible 
"""
from pathlib import Path

import ase.io
import torch

from mace import modules
from mace.data.atom_data_loader import AtomDataLoaderBuilder
from mace.training import Trainer, initialise_autoencoder


# File paths
TUTORIAL_ROOT = Path(__file__).resolve().parent
XYZ_FILE = TUTORIAL_ROOT / "data" / "xyz" / "A01_ethene_grid_static.xyz"
OUTPUT_DIR = TUTORIAL_ROOT / "_outputs" / "sample_script"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Based on the github, uses torch float 32
    torch.set_default_dtype(torch.float2)  
    torch.manual_seed(42)

    atoms = ase.io.read(XYZ_FILE, index=":")

    # Update here: Can now specify the E0s inside data_builder
    # The E0s itself will be converted accordingly, if not provided then 
    # it will be calculated from the average by default 
    # Format: {"element": Energy in eV}
    data_builder = AtomDataLoaderBuilder(
        cutoff=5.0,
        energy_key="REF_energy",
        forces_key="REF_forces",
        E0s={"H": -15.829, "C": -1031.3} 
        # FYI these are just proxy values
        # Need to update to use the one Marvin calculated
    )
    # CLI claims to use batch size 10
    data_loader = data_builder.load(
        atoms,
        batch_size=10,
        shuffle=False,
        drop_last=False,
    )

    # For model we just use default ANI500K presets 
    # Slightly defers from the github defaults but bopes
    # For TL later on load_base now also accepts file path input
    model = initialise_autoencoder(
        data_builder.get_metadata(), preset="default_ani", load_base="ani500k"
    )

    # The loss weights here are legit a mystery because the CLI and paper and
    # Github all dont match ???
    # CLI: Energy 1, Forces 100, Dipoles 1, NACs 100, SOCs 100
    # Github (Default): Energy 100, Forces 100
    # Github (TL from foundation Model): Energy 100, Forces 1
    # Paper: "aF and aC were set to 100 if aC was included in training"
    # -> Means when train NAC then Forces set to 100?? 
    # So for now idk all just set to 1.0 and see how
    loss_fn = modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(
        energy_weight=1.0,
        forces_weight=1.0,
        dipoles_weight=0.0,
        nacs_weight=0.0,
        socs_weight=0.0
    ).to(DEVICE)
    trainer = Trainer(
        max_epochs=100, # Default from the github, hopefully its enough
        device=DEVICE,
        verbose=True,
        # Github says 1e-4 but Paper says 1e-3..?? But 1e-3 feels more right
        optimiser_lr=1.0e-3, 
        # New addition for Adam weight decay
        # Affects only selected interaction and product layers (about 50% of params)
        optimiser_weight_decay=5.0e-7, 
        # Gradient clipping
        max_grad_norm=10.0,
        # New addition for scheduler, to decay the learning rate when validation
        # does not improve (patience is 50 here) 
        scheduler_lr_factor=0.8,
        scheduler_patience=50,
        # Ema now added, matches github, this affects validation metric
        ema_decay=0.99,
        early_stopping=True,
        stopping_patience=2048, # This means early stopping is technically disabled
        # Restores to the best ema validation loss
        restore_best=True
    )

    # Try train k fold
    models, full_history = trainer.train_k_fold_models(
        model,
        data_loader,
        loss_fn,
        k=5,
        seed=42,
    )

    # K-fold models are returned on CPU so they remain portable when saved.
    for model_name, fold_model in models.items():
        torch.save(fold_model, OUTPUT_DIR / f"{model_name}.model")
    torch.save(full_history, OUTPUT_DIR / "kfold_history.pt")

    # Output the metrics 
    print(f"Saved k-fold outputs to: {OUTPUT_DIR}")
    print("Combined k-fold metrics:")
    for metric, (mean, variance) in full_history["combined"].items():
        print(f"{metric}: mean={mean:.6f}, variance={variance:.6f}")


if __name__ == "__main__":
    main()
