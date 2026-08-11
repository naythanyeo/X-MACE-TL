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
    - metadata accessible through get_metadata(), but stored in class

Target API
atoms_list = ase.io.read(XYZ_FILE, index=f":{N_GEOMETRIES}")

# split atoms_list into train / valid / test
train_list, valid_list = train_test_split(atoms_list, 0.8)

data_builder = AtomDataLoaderBuilder(cutoff = 5, # Maximum bond length
                                     energy_key = "REF_energy",
                                     forces_key = "REF_forces")

train_loader = data_builder.load(train_list)
valid_loader = data_builder.load(valid_list)

data_metadata = data_builder.get_metadata()

# Initialised only with new non default parameters + data_metadata
model = initialise_base_autoencoder(latent=16... , data_metadata)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols

from mace.tools import AtomicNumberTable, get_atomic_number_table_from_zs
from mace.tools.torch_geometric import DataLoader

from .atomic_data import AtomicData
from .utils import compute_average_E0s, config_from_atoms_list


@dataclass
class AtomDataMetadata:
    """
    Metadata class that tracks all the info used to initialise a model
    Primary purpose is to store all these info that model requires for initialisation
    atomic_energies are stored and given to model as a E0_matrix for different heads
    """
    z_table: AtomicNumberTable
    atomic_energies: np.ndarray
    r_max: float
    n_energies: int
    avg_num_neighbors: float
    head_to_index: Dict[str, int]
    atomic_numbers: List[int] = field(init=False)
    num_elements: int = field(init=False)
    num_heads: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Other attributes used by model that need to be calculated
        """
        self.atomic_numbers = [int(z) for z in self.z_table.zs]
        self.num_elements = len(self.atomic_numbers)
        self.num_heads = len(self.head_to_index)


@dataclass
class AtomDataLoaderBuilder:
    """
    Initialise this builder class with deafult parameters

    Later on when creating load, accepted input forms for data:
     - List[Atoms] for ordinary single-head data.
     - Dict[str, List[Atoms]] for labelled multi-head data.

    """
    cutoff: float = 5.0  # Max bond length
    # xyz file labels (ignore nac and socs)
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    E0s: Optional[
        Union[Dict[str, float], Dict[str, Dict[str, float]]]
    ] = None
    _metadata: Optional[AtomDataMetadata] = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate the parameters
        """
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if not isinstance(self.energy_key, str) or not self.energy_key:
            raise ValueError("energy_key must be a non-empty string.")
        if not isinstance(self.forces_key, str) or not self.forces_key:
            raise ValueError("forces_key must be a non-empty string.")

        if self.E0s is not None:
            """
            If E0s is specified, we normalise it into a standard form
            Accept either a dictionary of E0s, or a dictionary of heads
            containing dictionaries of E0s
            If only a dictionary of E0s is specified, we normalise this to
            output "default": {Dict of E0s} so that the shapes are more
            consistent later on
            """
            if not isinstance(self.E0s, dict):
                raise ValueError(
                    "E0s should be a dictionary!!"
                )

            values = list(self.E0s.values())
            if all(not isinstance(value, dict) for value in values):
                self.E0s = {"default": self.E0s}
            elif not all(isinstance(value, dict) for value in values):
                raise ValueError(
                    "Each head E0s must be a dictionary"
                )

    def load(
        self,
        atoms,
        batch_size: int = 10,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Main loader function to convert an atoms list into dataloader
        1) Normalise list or dict of atoms into dict and validate it
        2) Build configuration objects by head
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

        # Construct the metadata
        if self._metadata is None:
            # First thing is to get the head to index mapping
            head_to_index = {
                name: index for index, name in enumerate(configs_by_head)
            }
            atomic_dataset = self._build_atomic_dataset(
                configs_by_head, z_table, head_to_index
            )
            self._metadata = self._build_metadata(
                configs_by_head, z_table, atomic_dataset, head_to_index
            )
        else:
            self._validate_metadata(z_table, configs_by_head)
            atomic_dataset = self._build_atomic_dataset(
                configs_by_head,
                self._metadata.z_table,
                self._metadata.head_to_index,
            )

        return DataLoader(
            dataset=atomic_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def get_metadata(self) -> AtomDataMetadata:
        if self._metadata is None:
            raise RuntimeError("Metadata is unavailable. Load training data first.")
        return self._metadata

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

    def _build_atomic_dataset(self, configs_by_head, z_table, head_to_index):
        """
        Convert the config objects into atomic_data objects
        Preserve each dictionary key as a numeric graph-level head
        """
        atomic_dataset = []
        for head_name, configs in configs_by_head.items():
            head_index = head_to_index[head_name]
            for config in configs:
                atomic_data = AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=self.cutoff,
                )
                atomic_data.head = torch.tensor(head_index, dtype=torch.long)
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

    @staticmethod
    def _convert_e0s_to_array(e0s, z_table) -> np.ndarray:
        """
        Helper to convert one E0s dictionary into an array of atomic energies
        ordered according to the z table
        Input: {"C": 5.0, "H", 0.4} for eg
        Output will be an array based on z tables
        This function is used in normalise e0s
        """
        symbols = [chemical_symbols[int(z)] for z in z_table.zs]
        if set(e0s) != set(symbols):
            raise ValueError(
                f"E0s must contain exactly these elements: {symbols}."
            )

        return np.array([e0s[symbol] for symbol in symbols], dtype=np.float64)


    def _normalise_e0s(self, configs_by_head, z_table, head_to_index):
        """
        Normalise supplied or calculated E0s into a dictionary of arrays
        Each dictionary keys are the heads, values are the arrays after they
        have been converted. Term this outputs as E0s by head
        """
        # First take the averages to get E0 if its not specified
        if self.E0s is None:
            e0s_by_head = {}
            for head_name, configs in configs_by_head.items():
                average_e0s = compute_average_E0s(configs, z_table)
                e0s_by_head[head_name] = np.array(
                    [average_e0s[z] for z in z_table.zs], dtype=np.float64
                )
            return e0s_by_head

        # If only one E0s are specified, but there are multiple heads, then
        # the E0s are duplicated to match the total number of heads too
        if set(self.E0s) == {"default"}:
            shared_e0s = self._convert_e0s_to_array(
                self.E0s["default"], z_table
            )
            return {
                head_name: shared_e0s.copy() for head_name in head_to_index
            }

        # If the specified E0s dont match the number of heads then raise error
        if set(self.E0s) != set(head_to_index):
            raise ValueError(
                "Head-specific E0s must contain exactly the configured head names."
            )
        # Convert the E0s values into the array format
        return {
            head_name: self._convert_e0s_to_array(
                self.E0s[head_name], z_table
            )
            for head_name in head_to_index
        }


    @staticmethod
    def _e0s_to_matrix(e0s_by_head, head_to_index) -> np.ndarray:
        """
        From the dictionary constructed, we convert it into a matrix
        This matrix is input into the model and accessed from there based
        on the head dimensions
        """
        ordered_heads = sorted(head_to_index, key=head_to_index.get)
        return np.stack([e0s_by_head[head] for head in ordered_heads])

    def _build_metadata(
        self, configs_by_head, z_table, atomic_dataset, head_to_index
    ) -> AtomDataMetadata:
        """
        Builds the important metadata from the data that is used to initialise
        the models later on. For eg, the z_tables and n_energies etc.
        This is stored in the metadata class so that model can easily reference it
        later when model is initialised
        """
        all_configs = self._flatten_configs(configs_by_head)
        n_energies = self._infer_n_energies(all_configs)
        e0s_by_head = self._normalise_e0s(
            configs_by_head, z_table, head_to_index
        )
        atomic_energies = self._e0s_to_matrix(e0s_by_head, head_to_index)
        avg_num_neighbors = self._compute_avg_num_neighbors(atomic_dataset)

        return AtomDataMetadata(
            z_table=z_table,
            atomic_energies=atomic_energies,
            r_max=self.cutoff,
            n_energies=n_energies,
            avg_num_neighbors=avg_num_neighbors,
            head_to_index=head_to_index,
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

    def _validate_metadata(self, z_table, configs_by_head) -> None:
        """
        Validate that the metadata used from before supports the data
        Check cutoff, z table, number of energy states and E0s if provided
        """
        if not np.isclose(self.cutoff, self._metadata.r_max):
            raise ValueError("cutoff must match metadata.r_max.")

        current_atomic_numbers = [int(z) for z in z_table.zs]
        if current_atomic_numbers != self._metadata.atomic_numbers:
            raise ValueError("Input z table must match metadata.z_table.")

        unknown_heads = set(configs_by_head) - set(self._metadata.head_to_index)
        if unknown_heads:
            raise ValueError(f"Input contains unknown heads: {sorted(unknown_heads)}.")

        all_configs = self._flatten_configs(configs_by_head)
        n_energies = self._infer_n_energies(all_configs)
        if n_energies != self._metadata.n_energies:
            raise ValueError(
                "Input data must contain the same number of energies as metadata."
            )

        # Convert E0s to array and double check for consistency if both 
        # E0s and metadat are provided
        if self.E0s is not None:
            e0s_by_head = self._normalise_e0s(
                configs_by_head, z_table, self._metadata.head_to_index
            )
            supplied_e0s = self._e0s_to_matrix(
                e0s_by_head, self._metadata.head_to_index
            )
            if not np.allclose(supplied_e0s, self._metadata.atomic_energies):
                raise ValueError("Input E0s must match metadata.atomic_energies.")
