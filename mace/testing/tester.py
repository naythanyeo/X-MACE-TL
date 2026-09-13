"""
Helper class to do testing on the model 
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from mace.tools.torch_geometric import DataLoader

from .metrics import mean_absolute_error, phase_mean_absolute_error


@dataclass
class Tester:
    """
    Class mainly just stores functions to evaluate the results
    Initialised only with device type for now 
    """

    device: Union[str, torch.device] = "cpu"

    def run_test(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        compute_nacs: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """
        Run inference and store predictions and references from the test as self 
        attributes. Later on class functions can access those attributes and check error.
        
        """
        model.to(self.device)
        model.eval() # Put in eval mode (technically for X-MACE so far it does nth)

        pred_energies = []
        ref_energies = []
        pred_forces = []
        ref_forces = []
        pred_nacs = []
        ref_nacs = []
        pred_smooth_nacs = []
        ref_smooth_nacs = []
        nac_ptr = [0]

        # Loop through all the batches to get combined preds
        # NAC pointer is stored because here we group all predictions from every batch
        for batch in test_loader:
            batch = batch.to(self.device)
            output = model(batch.to_dict(), training=False, compute_force=True)

            pred_energies.append(output["energy"].detach().cpu())
            ref_energies.append(batch.energy.detach().cpu())
            pred_forces.append(output["forces"].detach().cpu())
            ref_forces.append(batch.forces.detach().cpu())
            if compute_nacs:
                pred_nacs.append(output["nacs"].detach().cpu())
                ref_nacs.append(batch.nacs.detach().cpu())
                pred_smooth_nacs.append(output["smooth_nacs"].detach().cpu())
                ref_smooth_nacs.append(batch.smooth_nacs.detach().cpu())
                batch_ptr = batch.ptr.detach().cpu().numpy()
                nac_ptr.extend((nac_ptr[-1] + batch_ptr[1:]).tolist())

        self.pred_energies = torch.cat(pred_energies).numpy()
        self.ref_energies = torch.cat(ref_energies).numpy()
        self.pred_forces = torch.cat(pred_forces).numpy()
        self.ref_forces = torch.cat(ref_forces).numpy()

        if compute_nacs:
            self.pred_nacs = torch.cat(pred_nacs).numpy()
            self.ref_nacs = torch.cat(ref_nacs).numpy()
            self.pred_smooth_nacs = torch.cat(pred_smooth_nacs).numpy()
            self.ref_smooth_nacs = torch.cat(ref_smooth_nacs).numpy()
            self.nac_ptr = np.asarray(nac_ptr, dtype=np.int64)
            return self.pred_energies, self.pred_forces, self.pred_nacs, self.pred_smooth_nacs

        self.pred_nacs = None
        self.ref_nacs = None
        self.pred_smooth_nacs = None
        self.ref_smooth_nacs = None
        self.nac_ptr = None
        return self.pred_energies, self.pred_forces
    """
    Energies: (N_geoms, N_states)
    Forces: (N_geoms, N_states, 3N_atoms)    
    For energy and force MAE by states, averaging is done along the other axes
    """
    def get_energy_mae(self) -> float:
        return float(mean_absolute_error(self.pred_energies, self.ref_energies))

    def get_energy_mae_by_state(self) -> np.ndarray:
        return mean_absolute_error(
            self.pred_energies, self.ref_energies, axis=0
        )

    def get_force_mae(self) -> float:
        return float(mean_absolute_error(self.pred_forces, self.ref_forces))

    def get_force_mae_by_state(self) -> np.ndarray:
        return mean_absolute_error(
            self.pred_forces, self.ref_forces, axis=(0, 2)
        )

    def get_smooth_nac_phase_mae(self) -> float:
        return float(
            phase_mean_absolute_error(
                predicted=getattr(self, "pred_smooth_nacs", None),
                reference=getattr(self, "ref_smooth_nacs", None),
                ptr=self.nac_ptr,
                num_states=self.ref_energies.shape[-1],
                axis=None
            )
        )

    def get_smooth_nac_phase_mae_by_pair(self) -> float:
        return phase_mean_absolute_error(
            predicted=getattr(self, "pred_smooth_nacs", None),
            reference=getattr(self, "ref_smooth_nacs", None),
            ptr=self.nac_ptr,
            num_states=self.ref_energies.shape[-1],
            axis=(0, 2)
        )
    

    def get_raw_nac_phase_mae(self) -> float:
        return float(
            phase_mean_absolute_error(
                predicted=getattr(self, "pred_nacs", None),
                reference=getattr(self, "ref_nacs", None),
                ptr=self.nac_ptr,
                num_states=self.ref_energies.shape[-1],
                axis=None
            )
        )
    
    def get_raw_nac_phase_mae_by_pair(self) -> float:
        return phase_mean_absolute_error(
            predicted=getattr(self, "pred_nacs", None),
            reference=getattr(self, "ref_nacs", None),
            ptr=self.nac_ptr,
            num_states=self.ref_energies.shape[-1],
            axis=(0, 2)
        )
