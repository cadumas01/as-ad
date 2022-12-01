''' 
Practice putting as-ds model into dynamical system (matrix represenation).
See goodnotes "Econ Matrix Practice" for full details
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":

    # let each state vector, x_t = [inflation; output_gap_t]
    

    ##### DEFINING CONSTANTS #####

    # inflation sensitivity to demand conditions (v bar)
    v = 1

    # output gap to (deviation in target) interest rate sensitivity (b bar)
    b = 1

    # target inflation percentage
    inflation_target = 2 

    # in steady state, output gap = 0
    output_gap_target = 0 

    x_0 = np.array([inflation_target, output_gap_target])


    # Transition matrix
    B = np.array([[0, v],[-b, 0]])

  
    #### GENERAL MODEL - x_{t+1} as a function of x_{t} and shocks ####

    def get_x_t_next(x_t, shocks):
        I2 = np.identity(2)
        C = np.linalg.inv(I2 - B) 
        
        return (C @ np.array([[1,0],[0,0]]) @ x_t) + (C @ (shocks + np.array([0, b*inflation_target])))


    ### DEFINING SHOCKS - Can change over time ###

    n_time_steps = 25

    # function describing supply and demand shocks over time returns a vector of shocks  
    def get_shocks(t):

        # defaults

        # o bar
        supply_shock = 0

        # a bar
        demand_shock = 0

        if t < 5:
            supply_shock = 1
            demand_shock = 0

        elif t > 5 and t < 12:
            supply_shock = 0
            demand_shock = 0

        return np.array([supply_shock, demand_shock])


    #### RUNNING and ANIMATING MODEL ####

    # initial conidtion
    x_t = x_0
    x_t_next = None

    # some matrix where the ith column represents x_i
    x_t_series = np.reshape(x_t, (2,1))
    print(x_t_series)


    for t in range(n_time_steps):
        print(f"At time t = {t}:\n\tinflation = {x_t[0]}\n\toutput gap = {x_t[1]}\n")

        # set x_t <- x_t_next
        x_t = get_x_t_next(x_t, get_shocks(t))

        # append to storage matrix
        x_t_series = np.concatenate((x_t_series, np.reshape(x_t, (2,1))), axis=1)


    inflation_series = x_t_series[0]
    print("inflation series = ", inflation_series)

    output_gap_series = x_t_series[1]
    print("output_gap series = ", output_gap_series)

    # change color over time
    color_shift = np.linspace(.1, .9, n_time_steps)
    colors = [(1 - c, .5, c) for c in color_shift]

    # constant point sizes
    sizes = 25 * np.ones(n_time_steps)
 
    fig = plt.figure()
    time_label = fig.text(0.05, 0.95, '')

    plt.xticks([-4,-2,0,2,4])
    plt.yticks([-2,0,2,4,6])
    plt.vlines(x=output_gap_target, ymin=-2, ymax=6, linestyle="dashed", color=(.5,.5,.5))
    plt.hlines(y=inflation_target, xmin=-4, xmax=4, linestyle="dashed", color=(.5,.5,.5))


    plt.xlabel("Output Gap " +  r'$(\tilde{Y})$')
    plt.ylabel("Inflation\n" + r'$(\pi_{t})$', rotation=0)

    graph = plt.scatter([], [])
    plt.title("AS-DS Model over time with some shock")

    def animate(i):

        graph.set_offsets(np.vstack((output_gap_series[:i+1], inflation_series[:i+1])).T)
        graph.set_facecolors(colors[:i+1])
        graph.set_sizes(sizes[:i+1])
        time_label.set_text(f"time, t = {min(i, n_time_steps-1)}")
        return graph

    ani = FuncAnimation(fig, animate, repeat=False, interval=300)
   
    plt.show()




    
