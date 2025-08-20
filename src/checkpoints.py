import os
import torch
import pytorch_lightning as pl
from threading import Thread

class AsyncTrainableCheckpoint(pl.Callback):
    def __init__(self, dirpath="checkpoints", every_n_epochs=1, prefix="epoch"):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        os.makedirs(dirpath, exist_ok=True)
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        
        trainable_state = {
            k: v.cpu() for k, v in pl_module.state_dict().items()
            if "image_model" not in k
        }

        filename = os.path.join(
            self.dirpath, f"{self.prefix}_{epoch+1}.ckpt"
        )

        # Save asynchronously
        def _save():
            torch.save(trainable_state, filename)

        Thread(target=_save, daemon=True).start()

    @staticmethod
    def load_trainable_checkpoint(model, ckpt_path, strict=True):
        """
        Loads trainable parameters from a checkpoint into a model.
        Only updates matching keys; frozen backbone remains untouched.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = model.state_dict()
        
        # Filter only keys that exist in the model
        filtered_ckpt = {k: v for k, v in checkpoint.items() if k in state_dict}
        
        # Load parameters
        state_dict.update(filtered_ckpt)
        model.load_state_dict(state_dict, strict=strict)
        return model
