"""Minimal ANI500k-initialized X-MACE training example."""

from pathlib import Path

import ase.io
import torch
from sklearn.model_selection import train_test_split

from mace import modules
from mace.data.atom_data_loader import AtomDataLoaderBuilder
from mace.training import Trainer, initialise_autoencoder


# File paths
TUTORIAL_ROOT = Path(__file__).resolve().parent
XYZ_FILE = TUTORIAL_ROOT / "data" / "xyz" / "A01_ethene_grid_static.xyz"
OUTPUT_DIR = TUTORIAL_ROOT / "_outputs" / "sample_script"
MODEL_FILE = OUTPUT_DIR / "ethene_ani500k_finetuned_cpu.model"

# Small example settings
N_GEOMETRIES = 100
VALID_FRACTION = 0.2
BATCH_SIZE = 16
MAX_EPOCHS = 2048
PATIENCE = 15
LEARNING_RATE = 1.0e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # ANI500k foundation weights use float64.
    torch.manual_seed(SEED)

    # Read a small H/C dataset and make an 80/20 train-validation split.
    atoms = ase.io.read(XYZ_FILE, index=f":{N_GEOMETRIES}")
    train_atoms, valid_atoms = train_test_split(
        atoms, test_size=VALID_FRACTION, random_state=SEED
    )

    # The builder creates graph batches and the metadata needed by the model.
    data_builder = AtomDataLoaderBuilder(
        cutoff=5.0, energy_key="REF_energy", forces_key="REF_forces"
    )
    train_loader = data_builder.load(train_atoms, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = data_builder.load(valid_atoms, batch_size=BATCH_SIZE, shuffle=False)

    # Build an ANI-compatible autoencoder and copy in the ANI500k backbone weights.
    model = initialise_autoencoder(
        data_builder.get_metadata(), preset="default_ani", load_base="ani500k"
    ).to(DEVICE)
    """
    Subsequently if training or testing out transfer learning, after training
    one base model, cannot use initialise_autoencoder because this load base
    only loads in the foundations (before latent space)
    If using the full model from before, instead use
    model = torch.load("filepath")
    """

    loss_fn = modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(
        energy_weight=1.0,
        forces_weight=5.0,
        dipoles_weight=0.0,
        nacs_weight=0.0,
        socs_weight=0.0,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        early_stopping=True,
        patience=PATIENCE,
        restore_best=True,
        device=DEVICE,
        verbose=True,
    )

    model, history = trainer.train_model(
        model, train_loader, valid_loader, optimizer, loss_fn
    )

    # Save the complete model on CPU so it can be loaded without a CUDA device.
    model = model.cpu()
    torch.save(model, MODEL_FILE)

    print(f"Best epoch: {history['best_epoch']}")
    print(f"Saved CPU model: {MODEL_FILE}")


if __name__ == "__main__":
    main()
