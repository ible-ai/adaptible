"""Adaptible - LLMs that can wander."""

from ._api import Adaptible
from ._api import ModelProtocol
from ._classes import InteractionHistory
from ._classes import InteractionRequest
from ._classes import InteractionResponse
from ._classes import ReviewResponse
from ._classes import SyncResponse
from ._classes import TrainingExample
from ._llm import StatefulLLM
from ._server import MutableHostedLLM
from .libs import revise
from .libs import InvalidRevisionError
from .libs import REWRITE_INSTRUCTIONS
