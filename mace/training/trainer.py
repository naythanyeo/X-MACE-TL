"""Reusable training loop for X-MACE models."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import torch

from mace.tools.torch_geometric import DataLoader


@dataclass
class Trainer:
    max_epochs: int
    early_stopping: bool = False
    patience: int = 10
    restore_best: bool = True
    max_grad_norm: Optional[float] = 10.0
    device: Union[str, torch.device] = "cpu"
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        if self.early_stopping and self.patience < 1:
            raise ValueError(
                "patience must be at least 1 when early stopping is enabled."
            )

        self.device = torch.device(self.device)

    def train_model(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ):
        model.to(self.device)

        history = {"epoch": [], "train_loss": [], "valid_loss": []}
        best_state = None
        best_epoch = 0
        best_valid_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(
                model, train_loader, optimizer, loss_fn, training=True
            )
            valid_loss = self._run_epoch(
                model, valid_loader, optimizer, loss_fn, training=False
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)

            # Now settle the early stopping logic 
            # If the validation loss doesnt decrease, add to counter
            # Save the best lowest validation loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Can toggle this to kill output
            if self.verbose:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                    f"valid_loss={valid_loss:.6f}"
                )

            # Only break if early stopping is true
            if self.early_stopping and patience_counter >= self.patience:
                break
        
        # If restore best, then go back to lowest validation loss state
        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)

        history["best_epoch"] = best_epoch
        history["best_valid_loss"] = best_valid_loss
        history["stopped_epoch"] = history["epoch"][-1]

        return model, history

    def _run_epoch(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        training: bool,
    ) -> float:
        """
        TBC one epoch run 

        Will need to add in the loss and autoencoder logic
        Probably modify the autoencoder loss first before adding this part 
        
        """
        raise NotImplementedError("_run_epoch() has not been implemented yet.")
