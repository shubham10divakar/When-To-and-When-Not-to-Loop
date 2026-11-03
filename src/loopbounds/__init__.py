"""When To and When Not to Loop — feasibility bounds for agentic LLM pipelines.

Layers:
  * ``loopbounds.model``   — L1 analytical model (closed forms + exact DP)
  * ``loopbounds.sim``     — L2 SimPy discrete-event simulation
  * ``loopbounds.harness`` — L3 instrumented real-run harness
"""

from loopbounds.params import Params, load_params, save_params

__all__ = ["Params", "load_params", "save_params"]
__version__ = "0.1.0"
