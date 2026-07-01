import experiment_tracker
import functions
import torch
from itertools import product


path = "results/"

number_of_functions = 5
degres = [10]
nums = [30,31,32] + list(range(1, 19))
layers_list = [3]
anzats_reps_list = [1,2,3]
qubits_list = [6]
Max_steps = 1000


for deg in degres:
    for _ in range(number_of_functions):
        # iterate over all combinations of circuit numbers, layers, ansatz repetitions, and qubits
        target_function = functions.function_to_learn(degree=deg)
        x = torch.linspace(-torch.pi, torch.pi, steps=800, requires_grad=False)
        target_y = target_function(x)
        for layer,anzats_reps,n_qubit,num in product(layers_list,anzats_reps_list,qubits_list,nums):
            experiment_id, final_weights, cost_history = experiment_tracker.train_and_record(x, target_y, 
                                                                                             circuit_num=num, n_qubits=n_qubit, 
                                                                                             layers=layer, anzats_reps=anzats_reps, 
                                                                                             path=path,
                                                                                             max_steps=Max_steps)
    print(f"Experiment completed for degree {deg}")
