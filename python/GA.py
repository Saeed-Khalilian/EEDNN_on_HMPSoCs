import random
from deap import base, creator, tools, algorithms
import argparse

def fitness_function(individual):

    # calculating the latency of sequence for each exit branch
    latency_e1=latencies[individual[0]][0]+ latencies[individual[1]][1]
    latency_e2=latencies[individual[0]][0]+ latencies[individual[2]][2]+ latencies[individual[3]][3]
    latency_e3=latencies[individual[0]][0]+ latencies[individual[2]][2]+ latencies[individual[4]][4]+ latencies[individual[5]][5]
    latency_e4=latencies[individual[0]][0]+ latencies[individual[2]][2]+ latencies[individual[4]][4]+ latencies[individual[6]][6]

    if (individual[1]==individual[2]):
          latency_e2+= latencies[individual[1]][1]
          latency_e3+= latencies[individual[1]][1]
          latency_e4+= latencies[individual[1]][1]
    if (individual[3]==individual[4]):
          latency_e3+= latencies[individual[3]][3]
          latency_e4+= latencies[individual[3]][3]
    if (individual[5]==individual[6]):
          latency_e4+= latencies[individual[5]][5]
    
    if individual[1]==individual[3] and latencies[individual[2]][2]<latencies[individual[1]][1]:
          latency_e2+= latencies[individual[1]][1]  - latencies[individual[2]][2]
          latency_e3+= latencies[individual[1]][1]  - latencies[individual[2]][2]
          latency_e4+= latencies[individual[1]][1]  - latencies[individual[2]][2]
    if individual[3]==individual[5] and latencies[individual[4]][4]<latencies[individual[3]][3]:
          latency_e3+= latencies[individual[3]][3] - latencies[individual[4]][4]
          latency_e4+= latencies[individual[3]][3] - latencies[individual[4]][4]

    # latency overhead for B processor if G and B are executed in parallel
    B_overhead= 0.6 
    if ('G' in individual) and ('B' in individual):
          indexes=[i for i in range(len(individual)) if individual[i]=='B']
          indexes2=[i for i in range(len(individual)) if individual[i]=='G']
          for b in indexes:
              if b==1 and (2 in indexes2):
                    latency_e1+=latencies['B'][1]* B_overhead
              elif b==3 and (4 in indexes2):
                    latency_e2+=latencies['B'][3 ]* B_overhead
              elif b==5 and (6 in indexes2):
                    latency_e3+=latencies['B'][5 ]* B_overhead
          for g in indexes2:
              if g==1 and (2 in indexes):
                    latency_e2+=latencies['B'][2] * B_overhead
                    latency_e3+=latencies['B'][2] * B_overhead
                    latency_e4+=latencies['B'][2] * B_overhead
              elif g==3 and (4 in indexes):
                    latency_e3+=latencies['B'][4] * B_overhead
                    latency_e4+=latencies['B'][4] * B_overhead
              elif g==5 and (6 in indexes):
                    latency_e4+=latencies['B'][6] * B_overhead

    # Add a small overhead to promote diverse processor usage and discourage repeated assignments
    if (individual[4]==individual[5]) :
        latency_e3+= 0.05
        latency_e4+= 0.05
    if  (individual[2]==individual[3] ):
        latency_e2+= 0.05
        latency_e3+= 0.05 
        latency_e4+= 0.05 
    if (individual[0]==individual[1]) :
        latency_e1+= 0.05 
        latency_e2+= 0.05 
        latency_e3+= 0.05 
        latency_e4+= 0.05 

    avg_latency=classifiers_utility[0] * latency_e1+  classifiers_utility[1] * latency_e2+  classifiers_utility[2] * latency_e3+ classifiers_utility[3] * latency_e4 
    worst_case_latency=  latency_e4

    if worst_case_latency> deadline:
        return (float('inf'),)

    return (avg_latency,)

def run_genetic_algorithm(pop_size=500, num_generations=250, crossover_prob=0.8, mutation_prob=0.5, top_k=10):
    
    # Generate an initial population of valid individuals
    population = [toolbox.individual() for _ in range(pop_size)]

    # Statistics to keep track of the progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(f[0] for f in x) / len(x))
    stats.register("min", lambda x: min(f[0] for f in x))

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations,stats=stats, verbose=True)
    
    def remove_duplicates(population):
        unique_population = []
        seen_individuals = set()
        for ind in population:
            # Convert the individual into a tuple or string to make it hashable
            individual_tuple = tuple(ind)  # or `''.join(ind)` if you prefer the string format
            if individual_tuple not in seen_individuals:
                unique_population.append(ind)
                seen_individuals.add(individual_tuple)
        return unique_population

    # Remove duplicates from the population
    population = remove_duplicates(population)

    top_individuals = tools.selBest(population, k=top_k)
    print("\nTop Best Individuals:")
    for i, ind in enumerate(top_individuals, 1):
        print(f"Fitness = {ind.fitness.values[0]}")
        print(f"Sequence = {''.join(ind)}")

        best_individual= ind

        latency_e1=latencies[best_individual[0]][0]+ latencies[best_individual[1]][1]
        latency_e2=latencies[best_individual[0]][0]+ latencies[best_individual[2]][2]+ latencies[best_individual[3]][3]
        latency_e3=latencies[best_individual[0]][0]+ latencies[best_individual[2]][2]+ latencies[best_individual[4]][4]+ latencies[best_individual[5]][5]
        latency_e4=latencies[best_individual[0]][0]+ latencies[best_individual[2]][2]+ latencies[best_individual[4]][4]+ latencies[best_individual[6]][6]
        total_samples= classifiers_utility[0]+classifiers_utility[1]+classifiers_utility[2]+classifiers_utility[3]
        avg_latency= (classifiers_utility[0] * latency_e1+  classifiers_utility[1] * latency_e2+  classifiers_utility[2] * latency_e3+ classifiers_utility[3] * latency_e4 )/ total_samples
        print("Best Avg Latency:", avg_latency, "Worst Case Latency:", latency_e4)
        print("-" * 40)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pop_size', type=int, default=500, help='Population size')
    parser.add_argument('--num_generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--crossover_prob', type=float, default=0.8, help='Crossover probability')
    parser.add_argument('--mutation_prob', type=float, default=0.5, help='Mutation probability')
    parser.add_argument('--top_k', type=int, default=15, help='Number of top individuals to show')
    parser.add_argument('--processes', nargs='+', default=["B", "L", "G"], help='List of process types')
    parser.add_argument('--deadline', type=float, default=40, help='Deadline for worst case latency')
    args = parser.parse_args()

    processes = args.processes
    deadline = args.deadline
    
    # Latency of each block and exit branch of EE ResNet18 on Rock Pi N10
    latencies={}
    latencies['B']=[8.09,8.15,6.60,4.77,9.57,3.15,11.18]
    latencies['L']=[12.03,11.63,7.21,6.83,7.25,4.29,10.70]
    latencies['G']=[8.57,5.29,7.39,4.04,7.96,3.07,11.43]
    classifiers_utility=[8222, 1015, 294 ,469 ] # validation with the conf of 0.6


    # Setting up the genetic algorithm with DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.choice, processes)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=7) #' n e.g., length of BGLBLBL'
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic algorithm operators
    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the genetic algorithm with the provided arguments
    run_genetic_algorithm(pop_size=args.pop_size, num_generations=args.num_generations, crossover_prob=args.crossover_prob, mutation_prob=args.mutation_prob, top_k=args.top_k)
 