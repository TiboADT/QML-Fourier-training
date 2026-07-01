import experiment_tracker
import functions
import torch
from itertools import product
import time


path = "results/"

number_of_functions = 5
degres = [10]
nums = list(range(1, 20)) +  [30,31,32]
layers_list = [3]
anzats_reps_list = [1,2,3]
qubits_list = [6]
Max_steps = 600


print(f"Starting experiments with degrees: {degres}, number of functions: {number_of_functions}, number of circuits: {len(nums)}")
# send the time and date to identify when the experiment started
print(f"Experiment started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

for deg in degres:
    for i in range(number_of_functions):
        print(f"--------------Function {i+1} of {number_of_functions} for degree {deg} Started--------------")
        # iterate over all combinations of circuit numbers, layers, ansatz repetitions, and qubits
        target_function = functions.function_to_learn(degree=deg)
        x = torch.linspace(-torch.pi, torch.pi, steps=800, requires_grad=False)
        with torch.no_grad():
            target_y = target_function(x)
        for layer,anzats_reps,n_qubit,num in product(layers_list,anzats_reps_list,qubits_list,nums):
            experiment_id, final_weights, cost_history = experiment_tracker.train_and_record(x, target_y, 
                                                                                             circuit_num=num, n_qubits=n_qubit, 
                                                                                             layers=layer, anzats_reps=anzats_reps, 
                                                                                             path=path,
                                                                                             max_steps=Max_steps)
            
    print(f"Experiment completed for degree {deg}")
