from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
