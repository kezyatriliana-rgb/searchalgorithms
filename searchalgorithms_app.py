from dataclasses import dataclass
import random
import streamlit as st

POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
GENERATIONS = 50
MAX_FITNESS = 80


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_fitness: int
    average_fitness: float
    best_ones: int


def fitness(chromosome: str) -> int:
    """Return 80 only when an 80-bit chromosome contains exactly 50 ones."""
    if len(chromosome) != CHROMOSOME_LENGTH or any(bit not in '01' for bit in chromosome):
        raise ValueError('Chromosome must contain exactly 80 binary digits.')
    return MAX_FITNESS - abs(TARGET_ONES - chromosome.count('1'))


def create_chromosome(rng: random.Random) -> str:
    return ''.join(rng.choice('01') for _ in range(CHROMOSOME_LENGTH))


def tournament_selection(population: list[str], rng: random.Random, size: int = 3) -> str:
    return max(rng.sample(population, k=size), key=fitness)


def crossover(parent1: str, parent2: str, rng: random.Random, rate: float) -> tuple[str, str]:
    if rng.random() >= rate:
        return parent1, parent2
    point = rng.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


def mutate(chromosome: str, rng: random.Random, rate: float) -> str:
    return ''.join(('1' if bit == '0' else '0') if rng.random() < rate else bit for bit in chromosome)


def run_ga(seed: int = 42, crossover_rate: float = 0.8, mutation_rate: float = 0.01) -> dict:
    """Evaluate exactly 50 generations using elitism."""
    if not 0 <= crossover_rate <= 1 or not 0 <= mutation_rate <= 1:
        raise ValueError('Rates must be between 0 and 1.')
    rng = random.Random(seed)
    population = [create_chromosome(rng) for _ in range(POPULATION_SIZE)]
    history: list[GenerationStats] = []
    best_ever = max(population, key=fitness)

    for generation in range(1, GENERATIONS + 1):
        current_best = max(population, key=fitness)
        if fitness(current_best) > fitness(best_ever):
            best_ever = current_best
        scores = [fitness(chromosome) for chromosome in population]
        history.append(GenerationStats(generation, fitness(best_ever), sum(scores) / len(scores), best_ever.count('1')))
        if generation == GENERATIONS:
            break
        next_population = [best_ever]
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, rng)
            p2 = tournament_selection(population, rng)
            c1, c2 = crossover(p1, p2, rng, crossover_rate)
            next_population.append(mutate(c1, rng, mutation_rate))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutate(c2, rng, mutation_rate))
        population = next_population

    return {'best_chromosome': best_ever, 'best_fitness': fitness(best_ever),
            'number_of_ones': best_ever.count('1'), 'history': history,
            'population_size': POPULATION_SIZE, 'chromosome_length': CHROMOSOME_LENGTH,
            'generations': GENERATIONS, 'seed': seed}


def main() -> None:
    st.set_page_config(page_title='Genetic Algorithm | BSD3513', page_icon='🧬', layout='wide')
    st.title('Genetic Algorithm: Binary Pattern Optimization')
    st.write('BSD3513 Introduction to Artificial Intelligence — Lab Report 2')
    st.info('Goal: find an 80-bit chromosome containing exactly 50 ones, with maximum fitness 80.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Population', POPULATION_SIZE)
    c2.metric('Chromosome Length', f'{CHROMOSOME_LENGTH} bits')
    c3.metric('Target Ones', TARGET_ONES)
    c4.metric('Generations', GENERATIONS)

    with st.sidebar:
        st.header('Experiment settings')
        seed = st.number_input('Random seed', min_value=0, value=42, step=1)
        crossover_rate = st.slider('Crossover rate', 0.0, 1.0, 0.8, 0.05)
        mutation_rate = st.slider('Mutation rate per bit', 0.0, 0.2, 0.01, 0.005)
        st.caption('Assignment parameters are fixed; only experiment settings can change.')

    if st.button('Run Genetic Algorithm', type='primary'):
        st.session_state['result'] = run_ga(int(seed), crossover_rate, mutation_rate)

    if 'result' in st.session_state:
        result = st.session_state['result']
        st.subheader('Results')
        a, b, c = st.columns(3)
        a.metric('Best Fitness', f"{result['best_fitness']}/{MAX_FITNESS}")
        b.metric('Number of Ones', result['number_of_ones'])
        c.metric('Target Achieved', 'Yes' if result['number_of_ones'] == TARGET_ONES else 'No')
        st.code(result['best_chromosome'], language=None)
        st.caption(f"Chromosome length: {len(result['best_chromosome'])} bits | "
                   f"Ones: {result['number_of_ones']} | Zeros: {CHROMOSOME_LENGTH - result['number_of_ones']}")
        st.subheader('Fitness across 50 generations')
        chart_data = [{'Generation': s.generation, 'Best Fitness': s.best_fitness,
                       'Average Fitness': s.average_fitness} for s in result['history']]
        st.line_chart(chart_data, x='Generation', y=['Best Fitness', 'Average Fitness'])
        with st.expander('Generation-by-generation data'):
            st.dataframe(chart_data, use_container_width=True)
    else:
        st.write('Click **Run Genetic Algorithm** to start the experiment.')

    st.subheader('Methodology')
    st.markdown('''
1. **Initialization:** Generate 300 random 80-bit chromosomes.
2. **Fitness:** Calculate `80 - abs(50 - number_of_ones)`.
3. **Selection:** Use tournament selection (three candidates).
4. **Crossover:** Apply single-point crossover.
5. **Mutation:** Independently flip bits with the specified probability.
6. **Elitism:** Preserve the best chromosome.
7. **Termination:** Evaluate exactly 50 generations.

**Complexity:** Approximately **O(G × P × L)** time and **O(P × L)** population memory,
where *G* is generations, *P* is population size, and *L* is chromosome length.
The algorithm is stochastic, so it does not guarantee the optimum in every run.
''')


if __name__ == '__main__':
    main()
