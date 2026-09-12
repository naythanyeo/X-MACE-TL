###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch
from mace.modules.nac_utils import align_batch_nacs


def mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    return torch.mean(torch.square((ref["energy"] - pred["energy"])))

def mean_squared_error_invariants(ref: Batch, pred: TensorDict) -> torch.Tensor:
    #print(1-weighted_split_diff.unsqueeze(-1))
    return torch.mean(torch.square((pred["encoded_energy"] - pred["invariant_vals"])))

def reconstruction_error_invariants(ref: Batch, pred: TensorDict) -> torch.Tensor:
    return torch.mean(torch.square(ref["centered_energy_difference"] - pred["centered_decoded_energy"]))

def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    configs_energy_weight = ref.energy_weight  # [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )  # []


def weighted_mean_squared_stress(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)  # [n_graphs, ]
    return torch.mean(
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )  # []

def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    configs_energy_weight = ref.energy_weight  # [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]

    energy_loss = torch.mean(
        configs_weight.unsqueeze(-1)
        * configs_energy_weight.unsqueeze(-1)
        * torch.sum(torch.square((ref["energy"] - pred["energy"])) / num_atoms.unsqueeze(-1), dim=-1)
    )

    return energy_loss

def weighted_mean_squared_virials(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)  # [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )  # []

def old_phase_rmse_loss(ref: Batch, pred: TensorDict) -> torch.Tensor:
    """
    OLD NAC LOSS FUNCTION
    OUTDATED, BAD
    """
    # nacs: [n_pairs, 3]
    neg = torch.sum(torch.square(ref["nacs"] - pred["nacs"]), dim=-1)  # ||y - ŷ||^2 per pair
    pos = torch.sum(torch.square(ref["nacs"] + pred["nacs"]), dim=-1)  # ||y + ŷ||^2 per pair
    err2 = torch.minimum(pos, neg)                                     # phase-invariant per pair
    return torch.sqrt(torch.mean(err2))    


def phase_rmse_loss(ref: Batch, pred: TensorDict) -> torch.Tensor:
    """
    Updated NACs RMSE loss function
    For NACs, the loss we use smooth NACs
    """
    nac_residue = align_batch_nacs(
        pred=pred["smooth_nacs"],
        ref=ref["smooth_nacs"],
        ptr=ref.ptr,
        num_states=ref["energy"].shape[-1]
    )
    return torch.sqrt(torch.mean(nac_residue.square()))

def mean_squared_error_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # forces: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1).unsqueeze(
        -1
    )
    return torch.mean(
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )  # []


def weighted_mean_squared_error_dipole(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dipole: [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)  # [n_graphs,1]
    return torch.mean(torch.square((ref["dipoles"] - pred["dipoles"]) / num_atoms))  # []

def phase_rmse_socs(ref: Batch, pred: TensorDict) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_socs_weight = torch.repeat_interleave(
        ref.nacs_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )
    neg = torch.sum(torch.square(ref["socs"] - pred["socs"]), dim=-1)
    pos = torch.sum(torch.square(ref["socs"] + pred["socs"]), dim=-1)
    err2 = torch.minimum(pos, neg)                  
    return torch.sqrt(torch.mean(err2))  

def conditional_mse_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # forces: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]

    # Define the multiplication factors for each condition
    factors = torch.tensor([1.0, 0.7, 0.4, 0.1])

    # Apply multiplication factors based on conditions
    c1 = torch.norm(ref["forces"], dim=-1) < 100
    c2 = (torch.norm(ref["forces"], dim=-1) >= 100) & (
        torch.norm(ref["forces"], dim=-1) < 200
    )
    c3 = (torch.norm(ref["forces"], dim=-1) >= 200) & (
        torch.norm(ref["forces"], dim=-1) < 300
    )

    err = ref["forces"] - pred["forces"]

    se = torch.zeros_like(err)

    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]

    return torch.mean(configs_weight * configs_forces_weight * se)


def conditional_huber_forces(
    ref: Batch, pred: TensorDict, huber_delta: float
) -> torch.Tensor:
    # Define the multiplication factors for each condition
    factors = huber_delta * torch.tensor([1.0, 0.7, 0.4, 0.1])

    # Apply multiplication factors based on conditions
    c1 = torch.norm(ref["forces"], dim=-1) < 100
    c2 = (torch.norm(ref["forces"], dim=-1) >= 100) & (
        torch.norm(ref["forces"], dim=-1) < 200
    )
    c3 = (torch.norm(ref["forces"], dim=-1) >= 200) & (
        torch.norm(ref["forces"], dim=-1) < 300
    )
    c4 = ~(c1 | c2 | c3)

    se = torch.zeros_like(pred["forces"])

    se[c1] = torch.nn.functional.huber_loss(
        ref["forces"][c1], pred["forces"][c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref["forces"][c2], pred["forces"][c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref["forces"][c3], pred["forces"][c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref["forces"][c4], pred["forces"][c4], reduction="none", delta=factors[3]
    )

    return torch.mean(se)


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.energy_weight * weighted_mean_squared_error_energy(
            ref, pred
        ) + self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.stress_weight * weighted_mean_squared_stress(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        return (
            self.energy_weight
            * self.huber_loss(ref["energy"] / num_atoms, pred["energy"] / num_atoms)
            + self.forces_weight * self.huber_loss(ref["forces"], pred["forces"])
            + self.stress_weight * self.huber_loss(ref["stress"], pred["stress"])
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        return (
            self.energy_weight
            * self.huber_loss(ref["energy"] / num_atoms, pred["energy"] / num_atoms)
            + self.forces_weight
            * conditional_huber_forces(ref, pred, huber_delta=self.huber_delta)
            + self.stress_weight * self.huber_loss(ref["stress"], pred["stress"])
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.virials_weight * weighted_mean_squared_virials(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100.0
        )  # multiply by 100 to have the right scale for the loss

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"dipole_weight={self.dipole_weight:.3f})"


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )

class WeightedEnergyForcesNacsDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipoles_weight=1.0, nacs_weight=1.0, socs_weight=10.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "nacs_weight",
            torch.tensor(nacs_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipoles_weight",
            torch.tensor(dipoles_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "socs_weight",
            torch.tensor(socs_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        loss = 0

        if ref["energy"].shape == pred["energy"].shape:
            loss = self.energy_weight * mean_squared_error_energy(ref, pred)

        if ref["forces"].shape == pred["forces"].shape:
            loss += self.forces_weight * mean_squared_error_forces(ref, pred)

        if ref["nacs"].shape == pred["nacs"].shape:
            loss += self.nacs_weight * phase_rmse_loss(ref, pred)
        
        if ref["socs"].shape == pred["socs"].shape:
            loss += self.socs_weight * phase_rmse_socs(ref, pred)

        if ref["dipoles"].shape == pred["dipoles"].shape:
            loss += self.dipoles_weight * weighted_mean_squared_error_dipole(ref, pred) * 100

        return loss


    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipoles_weight:.3f})"
        )

class InvariantsWeightedEnergyForcesNacsDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipoles_weight=1.0, nacs_weight=1.0, socs_weight=10.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "nacs_weight",
            torch.tensor(nacs_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipoles_weight",
            torch.tensor(dipoles_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "socs_weight",
            torch.tensor(socs_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        loss = 0
        if ref["energy"].shape == pred["energy"].shape:
            pred_energy_loss = mean_squared_error_energy(ref, pred)
            reconstructed_energy_loss = reconstruction_error_invariants(ref, pred)
            latent_space_alignment_loss = mean_squared_error_invariants(ref, pred)
            loss = self.energy_weight * (pred_energy_loss+reconstructed_energy_loss+latent_space_alignment_loss)
        else:
            pred_energy_loss, reconstructed_energy_loss, latent_space_alignment_loss = None, None, None
        
        if ref["forces"].shape == pred["forces"].shape:
            forces_loss = mean_squared_error_forces(ref, pred)
            loss += self.forces_weight * forces_loss
        else:
            forces_loss = None

        if ref["smooth_nacs"].shape == pred["smooth_nacs"].shape:
            smooth_nacs_loss = phase_rmse_loss(ref, pred)
            loss += self.nacs_weight * smooth_nacs_loss
        else:
            smooth_nacs_loss = None

        if ref["dipoles"].shape == pred["dipoles"].shape:
            dipole_loss = weighted_mean_squared_error_dipole(ref, pred) * 100
            loss += self.dipoles_weight * dipole_loss * 100
        else:
            dipole_loss = None

        if ref["socs"].shape == pred["socs"].shape:
            socs_loss = phase_rmse_socs(ref, pred)
            loss += self.socs_weight * socs_loss
        else:
            socs_loss = None
        
        loss_breakdown = {
            "pred_energy_loss": pred_energy_loss,
            "reconstructed_energy_loss": reconstructed_energy_loss,
            "latent_space_alignment_loss": latent_space_alignment_loss,
            "forces_loss": forces_loss,
            "smooth_nacs_loss": smooth_nacs_loss,
            "dipole_loss": dipole_loss,
            "socs_loss": socs_loss
        }

        self.loss_breakdown = loss_breakdown

        return loss


    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipoles_weight:.3f})"
        )
