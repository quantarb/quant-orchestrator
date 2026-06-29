__version__ = "0.1.0"

from quant_orchestrator.platforms.builtins import register_builtin_providers
from quant_orchestrator.pipeline import FunctionStage, Pipeline, PipelineContext

register_builtin_providers()

__all__ = ["FunctionStage", "Pipeline", "PipelineContext", "__version__"]
