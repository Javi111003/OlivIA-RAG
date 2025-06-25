from planner.entities import *

def evaluate_plan(plan: StudyPlan, student: Student, official_topics: Dict[str, Topic]) -> float:
    covered_topics = set()
    total_time = 0.0
    weakness_focus = 0.0
    difficulty_penalty = 0.0
    previous_difficulty = None

    for block in plan.blocks:
        topic = block.topic
        topic_name = topic.name

        if topic_name not in official_topics:
            continue

        covered_topics.add(topic_name)
        total_time += block.time_allocated

        current_mastery = student.topic_mastery.get(topic_name, 0.0)
        weakness_focus += block.time_allocated * (1 - current_mastery * 0.1)

        if previous_difficulty is not None:
            difficulty_penalty += abs(block.target_difficulty - previous_difficulty)
        previous_difficulty = block.target_difficulty

    # 1. Coverage
    required_topics = set(official_topics.keys())
    coverage = len(covered_topics & required_topics) / len(required_topics)

    # 2. Weakness focus
    normalized_focus = weakness_focus / max(1.0, total_time)

    # 3. Time efficiency
    if total_time <= plan.available_time:
        efficiency = 1.0
    else:
        excess = total_time - plan.available_time
        efficiency = 1 / (1 + excess)

    # 4. Smooth difficulty progression
    smoothness = 1 / (1 + difficulty_penalty)

    # 🎯 Final fitness (weights now add to 0.90)
    fitness = (
        0.25 * coverage +
        0.30 * normalized_focus +
        0.15 * efficiency +
        0.10 * smoothness
    )

    return fitness
