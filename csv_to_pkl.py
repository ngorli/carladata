########
# Converts the csv file data into a trajectory, then converts the trajectory into a .pkl file
#######

import numpy as np
import pickle
from imitation.data.type import Trajectory


def main():
    
    # open all of the files
    train_file = open("expert_examples/training.txt", "r")
    valid_file = open("expert_examples/valid.txt", "r")
    eval_file = open("expert_examples/eval.txt", "r")
    file_list = [train_file, valid_file, eval_file]
    
    for cur_file in file_list:
        # initialize values
        states = []
        actions = []

        # skip the header file
        next(cur_file)

        # split up every state action pair
        for line in cur_file:
            line = line.split()
            states.append(np.array(line[1:4]))
            actions.append(np.array(line[4:]))
        states.append(np.array(line[1:4]))
        
        print(states)
        print(actions)


        """       
        # add this trajectory
        trajectory_list.append(Trajectory(obs = np.array(states), acts = np.array(actions), terminal = True))
    
        # close the file
        cur_file.close()

    # create the .pkl files for the trajectories
    with open("expert_examples/training_data.pkl", "wb") as f:
        pickle.dump(trajectory_list[0], f)

    with open("expert_examples/validate_data.pkl", "wb") as f:
        pickle.dump(trajectory_list[1], f)

    with open("expert_examples/evaluate_data.pkl", "wb") as f:
        pickle.dump(trajectory_list[2], f)
        """

if __name__ == "__main__":
    main()




