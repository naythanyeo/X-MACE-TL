"""Reusable training loop for X-MACE models."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from torch_ema import ExponentialMovingAverage

from mace.modules.lora import has_lora_layers, merge_lora_weights
from mace.tools.torch_geometric import DataLoader

from .optimiser import build_optimiser


@dataclass
class Trainer:
    """
    General Trainer class that mostly follows the CLI default parameters

    Early stopping is implemented but by default CLI doesnt use it, and instead relies
    on LR reduction (patience 50) and max epoch 100 by default. If want to toggle manual
    early stopping, set early_stopping to True and set stopping_patience to a value less
    than max_epochs.

    scheduler_lr_factor and scheduler_patience are for the ReduceLROnPlateau scheduler.
    The optimiser and scheduler are defined before the epoch loop in train_model.

    EMA is default enabled to be 0.99 (as per their github)
    """
    max_epochs: int = 100
    device: Union[str, torch.device] = "cpu"
    verbose: bool = True

    optimiser_lr: float = 1e-3
    optimiser_weight_decay: float = 5e-7
    max_grad_norm: Optional[float] = 10.0

    scheduler_lr_factor: float = 0.8
    scheduler_patience: int = 50

    ema_decay: Optional[float] = 0.99

    early_stopping: bool = True
    stopping_patience: int = 2048
    restore_best: bool = True

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        if self.optimiser_lr <= 0.0:
            raise ValueError("optimiser_lr must be positive.")
        if self.optimiser_weight_decay < 0.0:
            raise ValueError("optimiser_weight_decay must be non-negative.")
        if not 0.0 < self.scheduler_lr_factor < 1.0:
            raise ValueError("scheduler_lr_factor must be between 0 and 1.")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience must be non-negative.")
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1, or None.")
        if self.early_stopping and self.stopping_patience < 1:
            raise ValueError(
                "stopping_patience must be at least 1 when early stopping is enabled."
            )

        self.device = torch.device(self.device)

    def train_model(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fn: torch.nn.Module,
        checkpoint_epoch: Optional[int] = None,
        checkpoint_models_dir=None,
    ):
        """
        Main trainer loop that controls the overall training like early stopping
        or restoring the previous states
        The training loop calls on _run_epoch using either training or validaiton mode 
        Learning rate scheduler and ema also controlled here 
        For ema, the validation context is defined and used during validation mode
        """
        if checkpoint_epoch is not None and (
            isinstance(checkpoint_epoch, bool)
            or not isinstance(checkpoint_epoch, int)
            or checkpoint_epoch < 1
        ):
            raise ValueError("checkpoint_epoch must be a positive integer or None.")

        if checkpoint_epoch is not None:
            if checkpoint_models_dir is None:
                raise ValueError(
                    "checkpoint_models_dir is required when checkpoint_epoch is set."
                )
            checkpoint_models_dir = Path(checkpoint_models_dir).expanduser().resolve()
            if checkpoint_models_dir.exists() and not checkpoint_models_dir.is_dir():
                raise NotADirectoryError(checkpoint_models_dir)
            checkpoint_models_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        model.to(self.device)
        optimiser = build_optimiser(
            model,
            lr=self.optimiser_lr,
            weight_decay=self.optimiser_weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimiser,
            factor=self.scheduler_lr_factor,
            patience=self.scheduler_patience
        )
        # Define EMA object once every training loop
        ema = None
        if self.ema_decay is not None:
            ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_decay)

        history = {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "valid_energy_mae": [],
            "valid_force_mae": [],
            "learning_rate": [],
            "checkpoint_models": [],
        }
        best_state = None
        best_epoch = 0
        best_valid_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            current_lr = optimiser.param_groups[0]["lr"]
            train_metrics = self._run_epoch(
                model, train_loader, optimiser, loss_fn, training=True, ema=ema
            )
            train_loss = train_metrics["loss"]

            # Use the ema context for validation loss
            validation_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            with validation_context:
                valid_metrics = self._run_epoch(
                    model, valid_loader, optimiser, loss_fn, training=False, test=True
                )
                valid_loss = valid_metrics["loss"]

                # Best copy depends on ema weights also
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if checkpoint_epoch is not None and epoch % checkpoint_epoch == 0:
                    checkpoint_path = (
                        checkpoint_models_dir
                        / f"checkpoint_epoch_{epoch:06d}.pt"
                    )
                    if checkpoint_path.exists():
                        raise FileExistsError(checkpoint_path)

                    if has_lora_layers(model):
                        checkpoint_model = merge_lora_weights(model, inplace=False)
                        torch.save(
                            checkpoint_model.state_dict(),
                            checkpoint_path,
                        )
                        del checkpoint_model
                    else:
                        torch.save(
                            model.state_dict(),
                            checkpoint_path,
                        )

                    history["checkpoint_models"].append(
                        {
                            "epoch": epoch,
                            "path": str(checkpoint_path.resolve()),
                        }
                    )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            history["valid_energy_mae"].append(valid_metrics["energy_mae"])
            history["valid_force_mae"].append(valid_metrics["force_mae"])
            history["learning_rate"].append(current_lr)

            # Update the optimiser learning rate for the next epoch
            scheduler.step(valid_loss)

            # Can toggle this to kill output
            if self.verbose:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                    f"valid_loss={valid_loss:.6f} | "
                    f"energy_mae={valid_metrics['energy_mae']:.6f} | "
                    f"force_mae={valid_metrics['force_mae']:.6f} | "
                    f"lr={current_lr:.2e}"
                )

            # Only break if early stopping is true
            if self.early_stopping and patience_counter >= self.stopping_patience:
                break
        
        # If restore best, then go back to lowest validation loss state
        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)

        if has_lora_layers(model):
            model = merge_lora_weights(model, inplace=True)

        history["best_epoch"] = best_epoch
        history["best_valid_loss"] = best_valid_loss
        history["stopped_epoch"] = history["epoch"][-1]

        return model, history

    def train_k_fold_models(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        k: int = 5,
        seed: int = 42,
        checkpoint_epoch: Optional[int] = None,
        checkpoint_models_dir=None,
    ):
        """
        Train k independent model copies using k-fold cross-validation.
        Records also the mean and variances of all of the k folds
        """
        dataset_size = len(data_loader.dataset)
        if not 2 <= k <= dataset_size:
            raise ValueError("k must be between 2 and the dataset size.")

        fold_loaders = self._build_fold_loaders(data_loader, k, seed)
        models = {}
        full_history = {}

        for fold, (train_loader, valid_loader) in enumerate(fold_loaders, start=1):
            if self.verbose:
                print(f"Fold {fold}/{k}")
            # Create a copy of the model to train
            fold_model = deepcopy(model).to(self.device)
            fold_checkpoint_dir = None
            if checkpoint_epoch is not None:
                fold_checkpoint_dir = (
                    Path(checkpoint_models_dir)
                    .expanduser()
                    .resolve()
                    / f"fold_{fold:02d}"
                )
            fold_model, fold_history = self.train_model(
                fold_model,
                train_loader,
                valid_loader,
                loss_fn,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_models_dir=fold_checkpoint_dir,
            )

            model_key = f"model_{fold}"
            models[model_key] = fold_model.cpu()
            full_history[model_key] = fold_history

        full_history["combined"] = self._combine_fold_histories(full_history)

        return models, full_history

    @staticmethod
    def _combine_fold_histories(full_history):
        """
        For each fold, record the various properties then take the mean and variance 
        Output into one combined dictionary 
        """

        fold_values = {
            "best_epoch": [],
            "valid_loss": [],
            "valid_energy_mae": [],
            "valid_force_mae": []
        }

        for history in full_history.values():
            best_epoch = history["best_epoch"]
            best_index = best_epoch - 1
            fold_values["best_epoch"].append(best_epoch)
            fold_values["valid_loss"].append(history["valid_loss"][best_index])
            fold_values["valid_energy_mae"].append(
                history["valid_energy_mae"][best_index]
            )
            fold_values["valid_force_mae"].append(
                history["valid_force_mae"][best_index]
            )

        combined = {}
        for metric, values in fold_values.items():
            values = torch.tensor(values, dtype=torch.float64)
            combined[metric] = (
                torch.mean(values).item(),
                torch.var(values, unbiased=False).item()
            )

        return combined

    @staticmethod
    def _build_fold_loaders(data_loader: DataLoader, k: int, seed: int):
        """
        Use SK-LEARN to build k fold loaders 
        """
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        loader_pairs = []
        for fold, (train_indices, valid_indices) in enumerate(
            splitter.split(range(len(data_loader.dataset)))
        ):
            train_loader = DataLoader(
                Subset(data_loader.dataset, train_indices.tolist()),
                batch_size=data_loader.batch_size,
                shuffle=True,
                drop_last=False,
                generator=torch.Generator().manual_seed(seed + fold)
            )
            valid_loader = DataLoader(
                Subset(data_loader.dataset, valid_indices.tolist()),
                batch_size=data_loader.batch_size,
                shuffle=False,
                drop_last=False
            )
            loader_pairs.append((train_loader, valid_loader))

        return loader_pairs

    def _run_epoch(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        training: bool, # Training vs Validation Mode
        test: bool = False,
        ema: Optional[ExponentialMovingAverage] = None
    ) -> dict:
        model.train(training)
        total_loss = 0.0
        num_batches = 0
        energy_absolute_error = None
        force_absolute_error = None
        energy_count = 0
        force_count = 0

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
                optimiser.zero_grad(set_to_none=True)

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

            if test:
                """
                Include test mode to record and store the MAE for energies and forces
                Test is only called during the validation steps
                """
                energy_error = torch.abs(output["energy"] - batch["energy"])
                force_error = torch.abs(output["forces"] - batch["forces"])
                batch_energy_error = energy_error.detach().sum()
                batch_force_error = force_error.detach().sum()

                if energy_absolute_error is None:
                    energy_absolute_error = batch_energy_error
                    force_absolute_error = batch_force_error
                else:
                    energy_absolute_error += batch_energy_error
                    force_absolute_error += batch_force_error

                energy_count += energy_error.numel()
                force_count += force_error.numel()

            if training:
                loss.backward()
                # Gradient clipping 
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.max_grad_norm
                    )
                optimiser.step()
                # Update the EMA weights after the optimiser step if EMA is enabled
                if ema is not None:
                    ema.update()

            total_loss += loss.detach().item()
            num_batches += 1

        if num_batches == 0:
            raise ValueError("DataLoader is empty.")

        metrics = {"loss": total_loss / num_batches}
        if test:
            metrics["energy_mae"] = energy_absolute_error.item() / energy_count
            metrics["force_mae"] = force_absolute_error.item() / force_count

        return metrics
