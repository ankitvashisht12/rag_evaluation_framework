from pydantic import BaseModel

class EvaluationConfig(BaseModel):
    experiment_prefix: str = ""
    description: str = ""
    max_concurrency: int = 4
    save_results: bool = False
    save_results_path: str = ""

    