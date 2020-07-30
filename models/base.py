from enum import Enum
from typing import List, Any, Union, Dict

import torch


class ModeSwitcherBase:
    """
    Allow instance.is_* and instance.with_* mode access
    i.e
    if pointer.is_argmax: print(instance.mode)
    or
    output_argmax, output_gradient = pointer.with_argmax(inp), pointer.with_gradient(inp)

    define modes as class ModeSwitcherBase.Mode:
    ```
    class Mode(ModeSwitcherBase.Mode):
        weights = "weights"
        position = "position"
        gradient = "gradient"
        argmax = "argmax"
    ```
    """

    class Mode(Enum):
        @classmethod
        def keys(cls):
            return list(cls.__members__.keys())

    def __getattr__(self, item):
        cleaned = item.replace("is_", "").replace("with_", "")
        if hasattr(self.Mode, cleaned):
            mode = getattr(self.Mode, cleaned)
            if "is_" in item:
                return self.mode == mode
            elif "with_" in item:
                self.mode = mode
                return self
        # noop
        return super().__getattr__(item)

    def __dir__(self):
        return list(super().__dir__()) + [f"is_{k}" for k in self.Mode.keys()] + [f"with_{k}" for k in self.Mode.keys()]

    def __str__(self):
        return f"{self.__class__.__name__}.with_{self.mode.name}"


class ExportImportMixin:
    """
    Simple wrapper to load / export model weights
    - allows to skip import of keys which contain a ignore tokens,
        i.e 'coder_w' will skip 'encoder_weights' and 'decoder_weights'
    """
    def load(self: torch.nn.Module, path, ignore: List[str] = None):
        model_dict = self.state_dict()
        pretrained_dict: Union[Dict, Any] = torch.load(path, map_location=next(self.parameters()).device)
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and (not ignore or all(i not in k for i in ignore))
        }

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    def export_model(self: torch.nn.Module, path):
        torch.save(self.state_dict(), path)
