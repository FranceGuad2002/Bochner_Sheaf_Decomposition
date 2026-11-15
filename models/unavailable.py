"""The joint sheaf diffusion models are not implemented in this public repository
(see Hernandez Caralt et al., 2024); asking for one raises an error."""

JOINT_SHEAF_MODELS = ('JointSheafParams', 'JointSheafParamsAlt', 'JointSheafVanilla')

_MESSAGE = (
    "{name} is not implemented in this public repository; see Hernandez Caralt et al. "
    "(2024) for details."
)


class JointModelUnavailable(NotImplementedError):
    """Raised whenever a joint sheaf diffusion model is requested."""

    def __init__(self, name):
        super().__init__(_MESSAGE.format(name=name))


def unavailable_model(name):
    """Return a stand-in class for model `name` that raises JointModelUnavailable on construction."""

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise JointModelUnavailable(name)

    _Unavailable.__name__ = _Unavailable.__qualname__ = f'{name}Unavailable'
    return _Unavailable
