from dataclasses import dataclass, field
from typing import List, Dict

# ---------------------
# 📘 ENTITIES
# ---------------------

@dataclass
class Student:
    topic_mastery: Dict[str, float]  # e.g., {"Algebra": 0.7, "Geometry": 0.4}
    target_score: float              # Target exam score

@dataclass
class Topic:
    name: str
    exam_weight: float              # Importance in the exam (0–1)
    base_difficulty: float          # Intrinsic difficulty (0–1)

@dataclass
class StudyBlock:
    topic: Topic
    time_allocated: float           # Hours allocated
    target_difficulty: float        # Desired difficulty level

@dataclass
class StudyPlan:
    blocks: List[StudyBlock] = field(default_factory=list)
    available_time: float = 0.0     # Total time available until the exam (hours)
