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


Current main dataclasses and workflow 

1. xyz_file --> read into list of ASE atoms objects 
    - This is the ideal entry point into the helper 

2. list of atoms objects -> config objects 
3. config objects -> atomic_data objects
    - list of atomic_data objects = dataset 
4. dataloader references atomic_data dataset

For multihead, implement head for this dataloader first 
Will need a batch sampler later on that uses these labels

Main class function
INPUT: 
    - list of atoms objects OR
    - dictionary of atoms objects
OUTPUT: 
    - dataloader object
    - metadata accessible through attribute, but stored in class
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms

from mace.tools import AtomicNumberTable, get_atomic_number_table_from_zs
from mace.tools.torch_geometric import DataLoader

from .atomic_data import AtomicData
from .utils import compute_average_E0s, config_from_atoms_list


@dataclass(frozen=True)
class AtomDataMetadata:
    z_table: AtomicNumberTable
    atomic_energies: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate input metadata
        """
        atomic_energies = np.asarray(self.atomic_energies, dtype=np.float64)
        if atomic_energies.ndim != 1:
            raise ValueError("atomic_energies must be a one-dimensional array.")
        if len(atomic_energies) != len(self.z_table):
            raise ValueError("atomic_energies must contain one value per element.")
        object.__setattr__(self, "atomic_energies", atomic_energies)

"""
Target API
atoms_list = ase.io.read(XYZ_FILE, index=f":{N_GEOMETRIES}")

# split atoms_list into train / valid / test
train_list, valid_list = torch.train_test_split(atoms_list, 0.8)

data_builder = AtomDataLoaderBuilder(cutoff = 5, # Maximum bond length
                                     energy_key = "REF_energy",
                                     force_key = "REF_forces")

train_loader = data_builder(train_list)
valid_loader = data_builder(valid_list)

data_metadata = data_builder.metadata

# Initialised only with new non default parameters + data_metadata
model = initialise_base_autoencoder(latent=16... , data_metadata)
"""
class AtomDataLoaderBuilder:
    """
    Accepted input forms:
     - List[Atoms] for ordinary single-head data.
     - Dict[str, List[Atoms]] reserved for future multi-head data.
    """
    def __init__(
        self,
        cutoff: float = 5.0,
        energy_key: str = "REF_energy",
        forces_key: str = "REF_forces",
        metadata: Optional[AtomDataMetadata] = None,
    ) -> None:
        self.cutoff = cutoff
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.metadata = metadata

    def load(
        self,
        atoms,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        atoms_by_head = self._normalise_atoms(atoms)
        atoms_list = next(iter(atoms_by_head.values()))
        configs = config_from_atoms_list(
            atoms_list,
            energy_key=self.energy_key,
            forces_key=self.forces_key,
        )

        if self.metadata is None:
            z_table = get_atomic_number_table_from_zs(
                z for config in configs for z in config.atomic_numbers
            )
            atomic_energies_dict = compute_average_E0s(configs, z_table)
            atomic_energies = np.array(
                [atomic_energies_dict[z] for z in z_table.zs], dtype=np.float64
            )
            self.metadata = AtomDataMetadata(
                z_table=z_table,
                atomic_energies=atomic_energies,
            )
        else:
            self._validate_elements(configs)

        atomic_dataset = [
            AtomicData.from_config(
                config,
                z_table=self.metadata.z_table,
                cutoff=self.cutoff,
            )
            for config in configs
        ]
        return DataLoader(
            dataset=atomic_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @staticmethod
    def _normalise_atoms(atoms) -> Dict[str, List[Atoms]]:
        if isinstance(atoms, list):
            atoms_by_head = {"default": atoms}
        elif isinstance(atoms, dict):
            atoms_by_head = atoms
        else:
            raise TypeError("atoms must be a list or dictionary of ASE Atoms objects.")

        if len(atoms_by_head) == 0:
            raise ValueError("atoms cannot be empty.")
        if len(atoms_by_head) > 1:
            raise NotImplementedError(
                "Multi-head inputs are not supported by this initial skeleton."
            )

        head, atoms_list = next(iter(atoms_by_head.items()))
        if not isinstance(head, str):
            raise TypeError("Dictionary head names must be strings.")
        if not isinstance(atoms_list, list):
            raise TypeError("Each dictionary value must be a list of ASE Atoms objects.")
        if len(atoms_list) == 0:
            raise ValueError("atoms cannot be empty.")
        if not all(isinstance(atom, Atoms) for atom in atoms_list):
            raise TypeError("Every dataset entry must be an ASE Atoms object.")

        return atoms_by_head

    def _validate_elements(self, configs) -> None:
        supported = set(self.metadata.z_table.zs)
        present = {
            int(z)
            for config in configs
            for z in config.atomic_numbers
        }
        unsupported = sorted(present - supported)
        if unsupported:
            raise ValueError(
                f"Input data contains elements not present in metadata: {unsupported}."
            )
