from entities import *
import random

def generate_random_plan(official_topics: Dict[str, Topic],
                         student: Student,
                         available_time: float,
                         min_blocks: int = 5,
                         max_blocks: int = 10,
                         min_time_per_block: float = 1.0,
                         max_time_per_block: float = 5.0) -> StudyPlan:
    
    # Elegir cuántos temas incluir en este plan
    num_blocks = random.randint(min_blocks, min(max_blocks, len(official_topics)))
    selected_topics = random.sample(list(official_topics.values()), num_blocks)
    random.shuffle(selected_topics)

    blocks = []
    total_allocated = 0.0

    for topic in selected_topics:
        # Tiempo aleatorio, sin exceder el tiempo restante
        max_time_this_block = min(max_time_per_block, available_time - total_allocated)
        if max_time_this_block < min_time_per_block:
            break  # no queda tiempo suficiente para otro bloque válido

        time_allocated = round(random.uniform(min_time_per_block, max_time_this_block), 2)
        total_allocated += time_allocated

        # Dificultad objetivo entre dificultad base y 1.0
        difficulty = round(random.uniform(topic.base_difficulty, 1.0), 2)

        block = StudyBlock(
            topic=topic,
            time_allocated=time_allocated,
            target_difficulty=difficulty
        )
        blocks.append(block)

    return StudyPlan(blocks=blocks, available_time=available_time)


def generate_population(pop_size: int,
                        official_topics: Dict[str, Topic],
                        student: Student,
                        available_time: float,
                        min_blocks: int = 5,
                        max_blocks: int = 10,
                        min_time_per_block: float = 1.0,
                        max_time_per_block: float = 5.0) -> List[StudyPlan]:
    """
    Genera una lista de planes de estudio aleatorios para usar como población inicial.
    """
    population = []
    for _ in range(pop_size):
        plan = generate_random_plan(
            official_topics=official_topics,
            student=student,
            available_time=available_time,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            min_time_per_block=min_time_per_block,
            max_time_per_block=max_time_per_block
        )
        population.append(plan)
    return population

def structured_tournament_selection(population: List[StudyPlan],
                                    fitness_fn) -> List[StudyPlan]:
    """
    Empareja aleatoriamente a los individuos de la población de dos en dos
    y selecciona el mejor de cada par.
    
    Devuelve una lista con los ganadores (mitad de la población original).
    """
    
    
    shuffled = population[:]
    random.shuffle(shuffled)
    
    winners = []

    if len(population) % 2 != 0:
        winners.append(population.pop())

    for i in range(0, len(shuffled), 2):
        plan1 = shuffled[i]
        plan2 = shuffled[i + 1]
        winner = max([plan1, plan2], key=fitness_fn)
        winners.append(winner)
    
    return winners

def order_crossover(parent1: StudyPlan, parent2: StudyPlan):
    """
    Order Crossover (OX) adaptado para StudyBlock: mantiene unicidad de temas.
    """
    len_blocks = min(len(parent1.blocks), len(parent2.blocks))
    if len_blocks < 2:
        return parent1, parent2  # no hay nada útil que cruzar

    # Elegimos un segmento de corte
    start, end = sorted(random.sample(range(len_blocks), 2))

    def ox(parent_a, parent_b):
        # Paso 1: copiar el segmento de parent_a
        segment = parent_a.blocks[start:end]
        segment_topics = {block.topic.name for block in segment}

        # Paso 2: rellenar con los bloques de parent_b que no están en el segmento
        remaining = [b for b in parent_b.blocks if b.topic.name not in segment_topics]

        # Paso 3: construir el hijo
        child_blocks = remaining[:start] + segment + remaining[start:]
        return StudyPlan(blocks=child_blocks, available_time=parent_a.available_time)

    child1 = ox(parent1, parent2)
    child2 = ox(parent2, parent1)
    return child1, child2

def mutate(plan: StudyPlan,
           mutation_rate: float = 0.3,
           time_shift_range: float = 1.0,
           difficulty_shift_range: float = 0.1) -> StudyPlan:
    """
    Aplica mutaciones aleatorias a un StudyPlan.

    - mutation_rate: probabilidad de aplicar una mutación a cada tipo.
    - time_shift_range: rango en horas para ajustar tiempos.
    - difficulty_shift_range: ajuste permitido a la dificultad.
    """
    new_blocks = [StudyBlock(
        topic=block.topic,
        time_allocated=block.time_allocated,
        target_difficulty=block.target_difficulty
    ) for block in plan.blocks]  # copia profunda de los bloques

    # --- Swap de bloques ---
    if len(new_blocks) >= 2 and random.random() < mutation_rate:
        i, j = random.sample(range(len(new_blocks)), 2)
        new_blocks[i], new_blocks[j] = new_blocks[j], new_blocks[i]

    # --- Ajustes individuales por bloque ---
    for block in new_blocks:
        # Mutación de tiempo
        if random.random() < mutation_rate:
            delta = random.uniform(-time_shift_range, time_shift_range)
            block.time_allocated = max(0.5, round(block.time_allocated + delta, 2))  # mínimo 0.5h

        # Mutación de dificultad
        if random.random() < mutation_rate:
            delta = random.uniform(-difficulty_shift_range, difficulty_shift_range)
            block.target_difficulty = max(
                block.topic.base_difficulty,
                min(1.0, round(block.target_difficulty + delta, 2))
            )

    return StudyPlan(blocks=new_blocks, available_time=plan.available_time)

def evolve_population(population: List[StudyPlan],
                      fitness_fn,
                      num_generations: int = 50,
                      mutation_rate: float = 0.3,
                      elitism: bool = True) :
    """
    Ejecuta el ciclo evolutivo completo.

    Returns:
        - La última población generada
        - El mejor plan encontrado durante la evolución
    """
    best_plan = max(population, key=fitness_fn)

    for generation in range(num_generations):
        # 1. Selección por torneo estructurado
        selected = structured_tournament_selection(population, fitness_fn)

        # 2. Cruce entre pares
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                child1, child2 = order_crossover(selected[i], selected[i + 1])
                offspring.extend([child1, child2])

        # 3. Mutación
        mutated_offspring = [mutate(child, mutation_rate=mutation_rate) for child in offspring]

        # 4. Evaluación del mejor
        if elitism:
            best_offspring = max(mutated_offspring, key=fitness_fn)
            if fitness_fn(best_offspring) > fitness_fn(best_plan):
                best_plan = best_offspring

            # Reemplazar al peor de la nueva población con el mejor actual
            worst_idx = min(range(len(mutated_offspring)), key=lambda i: fitness_fn(mutated_offspring[i]))
            mutated_offspring[worst_idx] = best_plan

        # 5. Avanzar a la siguiente generación
        population = mutated_offspring

    return population, best_plan
