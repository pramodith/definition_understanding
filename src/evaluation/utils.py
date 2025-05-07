from dataclasses import dataclass


@dataclass
class EvaluationModels:
    model_name: str
    judgellm_model_name: str
    is_local: bool
