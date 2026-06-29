__version__ = "0.1.0"

from quant_orchestrator.platforms.builtins import register_builtin_providers

register_builtin_providers()

__all__ = ["__version__"]
