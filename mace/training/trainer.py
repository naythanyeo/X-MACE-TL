"""Reusable training loop for X-MACE models."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import torch

from mace.tools.torch_geometric import DataLoader


@dataclass
class Trainer:
    """
    General Trainer class that mostly follows the CLI default parameters 
    Only patience here set to 50 with an early stopping boolean 
    Originally patience is set to 2048 (no early stopping)
    """
    max_epochs: int = 2048
    early_stopping: bool = True
    patience: int = 50
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
        """
        Main trainer loop that controls the overall training like early stopping
        or restoring the previous states. 
        """
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
        training: bool, # Training vs Validation Mode
    ) -> float:
        model.train(training)
        total_loss = 0.0
        num_batches = 0

        base_model = model.module if hasattr(model, "module") else model
        prepare_outputs = getattr(base_model, "prepare_loss_outputs", None)

        for batch in data_loader:
            """
            TBC for future multihead data loading then batch will contain additional
            labels that indicate which head it uses

            model and loss function will read the head label and then run the forward 
            or calculate weighted loss based on that 

            Trainer remains generic
            """
            batch = batch.to(self.device)
            batch_dict = batch.to_dict()

            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model(batch_dict, training=training)
            """
            Training = True or False puts the model into eval() or train() mode
            Normally this controls things like drop out but currently it does nothing
            because theres no dropout or batch norm 
            Importantly even for validation grad must remain True because its needed
            to calculate Forces via autograd
            """
            if prepare_outputs is not None:
                output = prepare_outputs(batch_dict, output)

            loss = loss_fn(pred=output, ref=batch)

            if training:
                loss.backward()
                # Gradient clipping 
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.max_grad_norm
                    )
                optimizer.step()

            total_loss += loss.detach().item()
            num_batches += 1

        return total_loss / num_batches
