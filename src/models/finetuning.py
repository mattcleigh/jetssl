from pytorch_lightning.callbacks import BaseFinetuning
from torch import nn


class CatchupToLR(BaseFinetuning):
    """Finetuning callback that has the backbone linearly catch up to the model."""

    def __init__(self, unfreeze_at_step: int = 1000, catchup_steps: int = 1000) -> None:
        super().__init__()
        self.unfreeze_at_step = unfreeze_at_step
        self.catchup_steps = catchup_steps
        self.steps_done = 0
        self.backbone_lr = 0
        self.frozen = True

    def on_fit_start(self, trainer, pl_module) -> None:
        """Raises
        MisconfigurationException:
            If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, nn.Module):
            return super().on_fit_start(trainer, pl_module)
        raise ValueError("The LightningModule should have a `backbone` attribute")

    def freeze_before_training(self, pl_module) -> None:
        """Prevent the backbone from training initially.

        Called before `configure_optimizers`
        """
        self.freeze(pl_module.backbone)

    def finetune_function(self, pl_module, step, optimizer) -> None:
        """Used to update the learning rate of the backbone."""
        # Still in the frozen stage
        if step < self.unfreeze_at_step:
            pass

        # Time to thaw, initial learning rate is negligable
        elif self.frozen:
            self.unfreeze_and_add_param_group(
                pl_module.backbone,
                optimizer,
                1e-8,  # Start with no learning rate
            )
            self.frozen = False

            # Add the group keys that are missing else many schedulers will fail
            original = optimizer.param_groups[0]
            for key in original:
                if key not in optimizer.param_groups[-1]:
                    optimizer.param_groups[-1][key] = original[key]

        # Linearly ramp up
        elif self.steps_done < self.catchup_steps:
            model_lr = optimizer.param_groups[0]["lr"]
            delta = model_lr - self.backbone_lr
            steps_left = self.catchup_steps - self.steps_done
            increment = delta / steps_left
            self.backbone_lr = min(model_lr, self.backbone_lr + increment)
            optimizer.param_groups[-1]["lr"] = self.backbone_lr
            self.steps_done += 1

        # Fully caught up, ensure learning rates are always synced
        else:
            optimizer.param_groups[-1]["lr"] = optimizer.param_groups[0]["lr"]

    def on_train_batch_end(
        self, trainer, pl_module, _outputs, _batch, _batch_idx
    ) -> None:
        """Update the earning rate of the group after each batch pass.

        Same function as the partents old 'on_train_epoch_start' and also using global
        step instead of epoch.
        """
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            num_param_groups = len(optimizer.param_groups)
            self.finetune_function(pl_module, trainer.global_step, optimizer)
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Called when the epoch begins.

        Overloaded from parent to prevent anything happening
        """
