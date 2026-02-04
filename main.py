import networkx as nx
import random
import time
import matplotlib.pyplot as plt


def add_diffusion_probabilities(graph, min_prob=0, max_prob=0.2):
    for u, v in graph.edges():
        graph[u][v]['diffusion_prob'] = random.uniform(min_prob, max_prob)


def plot_influence_vs_iterations(best_influences, cumulative_times):
    plt.figure(figsize=(30, 6))
    iterations = list(range(1, len(best_influences) + 1))
    plt.plot(iterations, best_influences, marker='o', linestyle='-', color='b')
    for i in range(1, len(best_influences) - 1):
        if (best_influences[i] < best_influences[i + 1] or best_influences[i - 1] < best_influences[i]):
            x = iterations[i]
            y = best_influences[i]
            t = cumulative_times[i]
            plt.annotate(f'{t:.1f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title('Best Influence Over Iterations with Time Annotations at Turning Points')
    plt.xlabel('Iteration Number')
    plt.ylabel('Best Influence')
    plt.grid(True)
    plt.show()


class ACO_IM:
    def __init__(self, graph, k, ants, steps, alpha, beta, rho, epsilon, q0):
        self.graph = graph
        self.k = k
        self.ants = ants
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.q0 = q0
        self.pheromones = {node: epsilon for node in self.graph.nodes()}
        self.best_set = None
        self.best_influence = 0

    def run(self):
        best_influences = []  # List to store the best influence of each iteration
        iteration_times = []  # List to store time taken for each iteration
        start_time = time.time()  # Record the start time of the algorithm
        for step in range(self.steps):
            all_solutions = []
            for ant in range(self.ants):
                solution = self.construct_solution(step)
                influence = self.simulate_influence_spread(solution)
                all_solutions.append((solution, influence))
                if influence > self.best_influence:
                    self.best_influence = influence
                    self.best_set = solution
            self.update_pheromones(all_solutions)
            iteration_end = time.time()  # End time of the iteration
            iteration_times.append(iteration_end - start_time)
            print(
                f"Iteration {step + 1}: Best Solution = {self.best_set}, Best Influence = {self.best_influence}, time = {iteration_end - start_time}")
            best_influences.append(self.best_influence)  # Store the best influence of this iteration
        return self.best_set, self.best_influence, best_influences, iteration_times

    def construct_solution(self, step):
        current_solution = set()
        while len(current_solution) < self.k:
            next_node = self.select_next_node(current_solution, step)
            current_solution.add(next_node)
        return current_solution

    def select_next_node(self, current_solution, step):
        unvisited = set(self.graph.nodes()) - current_solution
        alpha_t = self.get_alpha(step)
        if random.random() < self.q0:
            next_node = max(unvisited, key=lambda x: self.pheromones[x] ** alpha_t * self.heuristic(x) ** self.beta)
        else:
            weights = [self.pheromones[node] ** alpha_t * self.heuristic(node) ** self.beta for node in unvisited]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            next_node = random.choices(list(unvisited), weights=probabilities)[0]
        return next_node

    def heuristic(self, node):
        if isinstance(self.graph, nx.DiGraph):
            return self.graph.out_degree(node)
        else:
            return len(self.graph[node])

    def get_alpha(self, t):
        if t <= 100:
            return 1
        elif t <= 400:
            return 2
        elif t <= 800:
            return 3
        else:
            return 4

    def simulate_influence_spread(self, influencers):
        influenced = set(influencers)
        active = list(influencers)
        while active:
            new_active = set()
            for node in active:
                neighbors = set(self.graph[node]) - influenced
                for neighbor in neighbors:
                    if random.random() < self.graph[node][neighbor]['diffusion_prob']:
                        # Use edge-specific diffusion probability
                        new_active.add(neighbor)
            influenced.update(new_active)
            active = new_active
        return len(influenced)

    def update_pheromones(self, solutions):
        for node in self.pheromones:
            self.pheromones[node] *= (1 - self.rho)
        for solution, influence in solutions:
            for node in solution:
                self.pheromones[node] += self.rho * (influence / self.best_influence)


class ACO_IM_PSO(ACO_IM):
    def __init__(self, graph, k, ants, steps, alpha, beta, rho, epsilon, q0, c1, c2, c3, phi= 0.0002, tau_max=6,
                 tau_min=0.01):
        super().__init__(graph, k, ants, steps, alpha, beta, rho, epsilon, q0)
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.c3 = c3  # Weight of previous velocity
        self.phi = phi
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.velocity = {node: 0.0 for node in self.graph.nodes()}
        self.local_best_pheromone = self.pheromones.copy()
        self.global_best_pheromone = self.pheromones.copy()

    def update_pheromones_with_pso(self, solutions):
        for node in self.pheromones:
            self.pheromones[node] *= (1 - self.rho)
            self.pheromones[node] = max(self.tau_min, self.pheromones[node])
        for solution, influence in solutions:
            for node in solution:
                if influence > self.best_influence:
                    self.global_best_pheromone[node] = self.pheromones[node]
                if influence > self.local_best_pheromone[node]:
                    self.local_best_pheromone[node] = self.pheromones[node]

                r1, r2 = random.random(), random.random()
                new_velocity = (self.c1 * r1 * (self.local_best_pheromone[node] - self.pheromones[node]) +
                                self.c2 * r2 * (self.global_best_pheromone[node] - self.pheromones[node]) +
                                self.c3 * self.velocity[node])

                self.velocity[node] = new_velocity
                self.pheromones[node] += self.velocity[node]
                self.pheromones[node] = max(self.tau_min, min(self.tau_max, self.pheromones[node]))

    def run(self):
        best_influences = []
        iteration_times = []
        start_time = time.time()
        for step in range(self.steps):
            all_solutions = []
            for ant in range(self.ants):
                solution = self.construct_solution(step)
                influence = self.simulate_influence_spread(solution)
                all_solutions.append((solution, influence))
                if influence > self.best_influence:
                    self.best_influence = influence
                    self.best_set = solution
            self.update_pheromones_with_pso(all_solutions)
            self.update_rho()  # Update rho after each iteration
            iteration_end = time.time()
            iteration_times.append(iteration_end - start_time)
            print(
                f"Iteration {step + 1}: Best Solution = {self.best_set}, Best Influence = {self.best_influence}, time = {iteration_end - start_time}")
            best_influences.append(self.best_influence)
        return self.best_set, self.best_influence, best_influences, iteration_times

    def update_rho(self):
        if self.rho > 0.95:
            self.rho = 0.95
        else:
            self.rho *= (1 - self.phi)


def run_experiment(graph, algorithm_cls, **params):
    algorithm = algorithm_cls(graph=graph, **params)
    start_time = time.time()
    influencers, influence, best_influences, iteration_times = algorithm.run()
    elapsed_time = time.time() - start_time
    plot_influence_vs_iterations(best_influences, iteration_times)
    return influencers, influence, elapsed_time


def main():
    graph_path = 'facebook_combined.txt'
    G_graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph())

    # Add random diffusion probabilities to each edge
    add_diffusion_probabilities(G_graph)

    aco_params = {
        'k': 5,
        'ants': 30,
        'steps': 150,
        'alpha': 1,  # Initial value
        'beta': 1,
        'rho': 0.85,  # Initial value
        'epsilon': 0.01,
        'q0': 0.6
    }

    pso_aco_params = {
        'k': 5,
        'ants': 30,
        'steps': 150,
        'alpha': 1.0,  # Initial value
        'beta': 1,
        'rho': 0.95,  # Initial value
        'epsilon': 0.01,
        'q0': 0.6,
        'c1': 0.3,
        'c2': 0.7,
        'c3': 0.3
    }

    G_aco_results = run_experiment(G_graph, ACO_IM, **aco_params)
    print("soc-Epinions1.txt Graph - ACO-IM Results:")
    print(f"Best Influencers: {G_aco_results[0]}")
    print(f"Influence Spread: {G_aco_results[1]}")
    print(f"Computation Time: {G_aco_results[2]:.2f} seconds\n")
    G_pso_aco_results = run_experiment(G_graph, ACO_IM_PSO, **pso_aco_params)
    print("soc-Epinions1.txt Graph - ACO-PSO-IM Results:")
    print(f"Best Influencers: {G_pso_aco_results[0]}")
    print(f"Influence Spread: {G_pso_aco_results[1]}")
    print(f"Computation Time: {G_pso_aco_results[2]:.2f} seconds\n")


if __name__ == "__main__":
    main()

"""

"""