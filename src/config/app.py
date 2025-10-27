from __future__ import annotations
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Mapping

from .core import CoreSettings
from .ace import ACEParameters
from .voting import VotingParameters
from .updating import BeliefParameters

@dataclass(slots=True)
class AppConfig:
    core: CoreSettings = field(default_factory=CoreSettings)
    ace: ACEParameters = field(default_factory=ACEParameters)
    voting: VotingParameters = field(default_factory=VotingParameters)
    updating: BeliefParameters = field(default_factory=BeliefParameters)

def all_config() -> AppConfig:
    return AppConfig()