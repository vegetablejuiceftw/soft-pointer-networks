from enum import Enum


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
        return getattr(super(), item)

    def __dir__(self):
        return list(super().__dir__()) + [f"is_{k}" for k in self.Mode.keys()] + [f"with_{k}" for k in self.Mode.keys()]

    def __str__(self):
        return f"{self.__class__.__name__}.with_{self.mode.name}"
