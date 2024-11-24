import math
import matplotlib.pyplot as plt
import numpy as np

x0 = 1
v0 = 0
omega = 1.0
gamma = -0.1
dt = 0.01
max_time = 20
time = np.arange(0, max_time+dt, dt)

def euler():
    x = [x0]
    v = [v0]

    for step in range(1, len(time)):
        x.append(x[step-1] + v[step-1] * dt)
        v.append(v[step-1] - (2 * gamma * v[step-1] + (x[step-1] * omega ** 2)) * dt)

    return x

def exact():
    omega1 = math.sqrt((omega ** 2) - (gamma ** 2))

    exact_list = [math.exp(-gamma * time_step) * x0 * math.cos(omega1 * time_step)
              + (v0 + gamma * x0) / omega1 * math.sin(omega1 * time_step) for time_step in time]

    return exact_list

def main():

    euler_solution = euler()
    exact_solution = exact()

    plt.plot(time, euler_solution)
    plt.plot(time, exact_solution)
    plt.legend(["Решение методом Эйлера", "Точное решение"], loc="lower right")

    plt.show()

main()