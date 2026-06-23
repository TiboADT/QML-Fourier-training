import experiment_tracker
import functions
import torch

path = "results/"

number_of_functions = 10
degres = [6,8,10,12]
nums = list(range(1,20)) + [31,32]
layers_list = [2,3,4,5]
anzats_reps_list = [1,2,3]
qubits_list = [6]


for deg in degres:
    for _ in range(number_of_functions):
        for num,layer,anzats_reps,n_qubit in zip(nums,layers_list,anzats_reps_list,qubits_list):
            target_function = functions.function_to_learn(degree=deg)
            x = torch.linspace(-torch.pi, torch.pi, steps=800, requires_grad=False)
            target_y = target_function(x)
            experiment_id, final_weights, cost_history = experiment_tracker.train_and_record(x, target_y, 
                                                                                             circuit_num=num, n_qubits=n_qubit, 
                                                                                             layers=layer, anzats_reps=anzats_reps, 
                                                                                             path=path)
    print(f"Experiment completed for degree {deg}")
