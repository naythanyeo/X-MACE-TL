"""
Helper class to load in dataloader objects 

This class is built with the following intentions 
1. Public API 
    - Only input list of atoms objects
    - Settles the internal config / z tables / baseline E
    - Outputs dataloaders and 1 metadata dataclass only
2. Multi-headed label
    - Support dataloading for multiheaded training later on
    - Accept either list of atoms objects or dictionary 
    - Add on head labels to atomic_data 
    - ideally same class for both 

Main class function
INPUT: 
    - list of atoms objects OR
    - dictionary of atoms objects
OUTPUT: 
    - dataloader object
    - metadata accessible through attribute, but stored in class

Target API
atoms_list = ase.io.read(XYZ_FILE, index=f":{N_GEOMETRIES}")

# split atoms_list into train / valid / test
train_list, valid_list = train_test_split(atoms_list, 0.8)

data_builder = AtomDataLoaderBuilder(cutoff = 5, # Maximum bond length
                                     energy_key = "REF_energy",
                                     forces_key = "REF_forces")

train_loader = data_builder.load(train_list)
valid_loader = data_builder.load(valid_list)

data_metadata = data_builder.metadata

# Initialised only with new non default parameters + data_metadata
model = initialise_base_autoencoder(latent=16... , data_metadata)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from ase import Atoms

from mace.tools import AtomicNumberTable, get_atomic_number_table_from_zs
from mace.tools.torch_geometric import DataLoader

from .atomic_data import AtomicData
from .utils import compute_average_E0s, config_from_atoms_list


@dataclass
class AtomDataMetadata:
    z_table: AtomicNumberTable
    atomic_energies: np.ndarray
    r_max: float
    n_energies: int
    avg_num_neighbors: float
    atomic_numbers: List[int] = field(init=False)
    num_elements: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Other attributes used by model that need to be calculated
        """
        self.atomic_numbers = [int(z) for z in self.z_table.zs]
        self.num_elements = len(self.atomic_numbers)


@dataclass
class AtomDataLoaderBuilder:
    """
    Accepted input forms:
     - List[Atoms] for ordinary single-head data.
     - Dict[str, List[Atoms]] for labelled multi-head data.

    Initialise this builder class with deafult parameters
    """
    cutoff: float = 5.0  # Max bond length
    # xyz file labels (ignore nac and socs)
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    metadata: Optional[AtomDataMetadata] = None

    def __post_init__(self) -> None:
        """
        Validate the parameters
        Metadata provided should be of the metadata object
        """
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if not isinstance(self.energy_key, str) or not self.energy_key:
            raise ValueError("energy_key must be a non-empty string.")
        if not isinstance(self.forces_key, str) or not self.forces_key:
            raise ValueError("forces_key must be a non-empty string.")
        if self.metadata is not None and not isinstance(
            self.metadata, AtomDataMetadata
        ):
            raise TypeError("metadata must be an AtomDataMetadata object.")
        if self.metadata is not None and not np.isclose(
            self.cutoff, self.metadata.r_max
        ):
            raise ValueError("cutoff must match metadata.r_max.")

    def load(
        self,
        atoms,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Main loader function to convert an atoms list into dataloader
        1) Normalise list or dict of atoms into dict and validate it
        2) Build configuration objects
        3) Build z table and AtomicData objects
        4) Build or validate metadata
        """
        atoms_by_head = self._normalise_atoms(atoms)

        # Build configuration object for each head
        # Convert dict of atoms list to dict of config list
        configs_by_head = {head: self._atoms2config(atoms_list)
                           for head, atoms_list in atoms_by_head.items()}

        # Reformat into a single list of all configs for metadata building
        all_configs = self._flatten_configs(configs_by_head)

        # Build the z table explicitly for every dataset
        z_table = get_atomic_number_table_from_zs(
            z for config in all_configs for z in config.atomic_numbers
        )

        # Build AtomicData before creating or validating metadata
        atomic_dataset = self._build_atomic_dataset(
            configs_by_head, z_table=z_table
        )

        if self.metadata is None:
            self.metadata = self._build_metadata(
                all_configs, z_table, atomic_dataset
            )
        else:
            self._validate_metadata(z_table, all_configs)

        return DataLoader(
            dataset=atomic_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def get_metadata(self) -> AtomDataMetadata:
        if self.metadata is None:
            raise RuntimeError("Metadata is unavailable. Load training data first.")
        return self.metadata

    def _normalise_atoms(self, atoms) -> Dict[str, List[Atoms]]:
        """
        Normalise so that input is always a dict of head: atom_list
        """
        if isinstance(atoms, list):
            atoms_by_head = {"default": atoms}
        elif isinstance(atoms, dict):
            atoms_by_head = atoms
        else:
            raise TypeError("atoms must be a list or dictionary of ASE Atoms objects.")

        return atoms_by_head

    def _atoms2config(self, atoms_list) -> List[object]:
        """
        Mini helper function to convert atoms list into config list
        """
        config = config_from_atoms_list(atoms_list,
                                        energy_key=self.energy_key,
                                        forces_key=self.forces_key)
        return config

    def _build_atomic_dataset(self, configs_by_head, z_table):
        """
        Convert the config objects into atomic_data objects
        Preserve each dictionary key as a raw head_label string
        TBC when implemented multihead, might use head_index here
        """
        atomic_dataset = []
        for head, configs in configs_by_head.items():
            for config in configs:
                atomic_data = AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=self.cutoff,
                )
                atomic_data.head_label = head
                atomic_dataset.append(atomic_data)
        return atomic_dataset

    @staticmethod
    def _flatten_configs(configs_by_head):
        """
        Mini helper function to reformat dictionary
        """
        all_configs = []
        for configs in configs_by_head.values():
            all_configs.extend(configs)
        return all_configs

    def _build_metadata(
        self, all_configs, z_table, atomic_dataset
    ) -> AtomDataMetadata:
        """
        Builds the important metadata from the data that is used to initialise
        the models later on. For eg, the z_tables and n_energies etc.
        This is stored in the metadata class so that model can easily reference it
        later when model is initialised
        """
        n_energies = self._infer_n_energies(all_configs)
        atomic_energies_dict = compute_average_E0s(all_configs, z_table)
        atomic_energies = np.array(
            [atomic_energies_dict[z] for z in z_table.zs], dtype=np.float64
        )
        avg_num_neighbors = self._compute_avg_num_neighbors(atomic_dataset)

        return AtomDataMetadata(
            z_table=z_table,
            atomic_energies=atomic_energies,
            r_max=self.cutoff,
            n_energies=n_energies,
            avg_num_neighbors=avg_num_neighbors,
        )

    @staticmethod
    def _infer_n_energies(configs) -> int:
        energy_sizes = {int(np.asarray(config.energy).size) for config in configs}
        if len(energy_sizes) != 1:
            raise ValueError(
                "All configurations must contain the same number of energies."
            )
        return energy_sizes.pop()

    @staticmethod
    def _compute_avg_num_neighbors(atomic_dataset) -> float:
        neighbor_counts = []
        for atomic_data in atomic_dataset:
            receivers = atomic_data.edge_index[1]
            if receivers.numel() > 0:
                _, counts = torch.unique(receivers, return_counts=True)
                neighbor_counts.append(counts)

        if not neighbor_counts:
            raise ValueError(
                "Cannot calculate average neighbors from a dataset with no edges."
            )

        counts = torch.cat(neighbor_counts).type(torch.get_default_dtype())
        return float(torch.mean(counts).item())

    def _validate_metadata(self, z_table, all_configs) -> None:
        """
        Validate that the metadata used from before supports the data
        Check cutoff, z table, and number of energy states
        """
        if not np.isclose(self.cutoff, self.metadata.r_max):
            raise ValueError("cutoff must match metadata.r_max.")

        current_atomic_numbers = [int(z) for z in z_table.zs]
        if current_atomic_numbers != self.metadata.atomic_numbers:
            raise ValueError("Input z table must match metadata.z_table.")

        n_energies = self._infer_n_energies(all_configs)
        if n_energies != self.metadata.n_energies:
            raise ValueError(
                "Input data must contain the same number of energies as metadata."
            )
