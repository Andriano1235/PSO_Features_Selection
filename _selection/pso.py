import numpy as np
import time
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split

from _selection.__utils__._utilities_pso import *
from _selection.__utils__._transfer_functions_pso import get_trans_function


def PSO(num_agents, max_iter, train_data, test_data, train_label, test_label, obj_function=compute_fitness, trans_func_shape='s', save_conv_graph=False):

    # Particle Swarm Optimizer
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of particles                                           #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    short_name = 'PSO'
    agent_name = 'Particle'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_func_shape)

    # setting up the objectives
    weight_acc = None
    if (obj_function == compute_fitness):
        # weight_acc = float(
        #     input('Weight for the classification accuracy [0-1]: '))
        weight_acc = 0.9
    obj = (obj_function, weight_acc)
    # compute_accuracy is just compute_fitness with accuracy weight as 1
    compute_accuracy = (compute_fitness, 1)

    # initialize particles and Leader (the agent with the max fitness)
    particles = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)

    # initialize data class
    data = Data()

    data.train_X, data.val_X, data.train_Y, data.val_Y = train_data, test_data, train_label, test_label

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial particles
    particles, fitness = sort_agents(particles, obj, data)

    # start timer
    start_time = time.time()

    # initialize global and local best particles
    globalBestParticle = [0 for i in range(num_features)]
    globalBestFitness = float("-inf")
    localBestParticle = [
        [0 for i in range(num_features)] for j in range(num_agents)]
    localBestFitness = [float("-inf") for i in range(num_agents)]
    weight = 0.9
    weight_max = 0.9
    weight_min = 0.4
    velocity = [[0.0 for i in range(num_features)] for j in range(num_agents)]

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # update weight
        # weight = 1.0 - (iter_no / max_iter)
        if iter_no == 0:
            weight = weight
        else:
            weight = weight_max-((weight_max - weight_min)*iter_no / max_iter)

        # update the velocity
        for i in range(num_agents):
            for j in range(num_features):
                velocity[i][j] = (weight*velocity[i][j])
                r1, r2 = np.random.random(2)
                velocity[i][j] = velocity[i][j] + \
                    (r1 * (localBestParticle[i][j] - particles[i][j]))
                velocity[i][j] = velocity[i][j] + \
                    (r2 * (globalBestParticle[j] - particles[i][j]))

        # updating position of particles
        for i in range(num_agents):
            for j in range(num_features):
                trans_value = trans_function(velocity[i][j])
                if (np.random.random() < trans_value):
                    particles[i][j] = 1
                else:
                    particles[i][j] = 0

        # updating fitness of particles
        particles, fitness = sort_agents(particles, obj, data)
        display(particles, fitness, agent_name)

        # updating the global best and local best particles
        for i in range(num_agents):
            if fitness[i] > localBestFitness[i]:
                localBestFitness[i] = fitness[i]
                localBestParticle[i] = particles[i][:]

            if fitness[i] > globalBestFitness:
                globalBestFitness = fitness[i]
                globalBestParticle = particles[i][:]

        # update Leader (best agent)
        if globalBestFitness > Leader_fitness:
            Leader_agent = globalBestParticle.copy()
            Leader_fitness = globalBestFitness.copy()

        convergence_curve['fitness'][iter_no] = np.mean(fitness)

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(
        Leader_agent, compute_accuracy, data)
    particles, accuracy = sort_agents(particles, compute_accuracy, data)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name +
          ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    print('Leader ' + agent_name +
          ' Classification Accuracy : {}'.format(Leader_accuracy))
    print('\n================================================================================\n')

    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # plot convergence graph
    fig, axes = Conv_plot(convergence_curve)
    if (save_conv_graph):
        plt.savefig('convergence_graph_' + short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_particles = particles
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution
