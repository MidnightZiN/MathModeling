import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.widgets import Slider
import numpy as np

# Константы
k = 1.0  # Жёсткость пружины (N/m)
m = 1.0  # Масса (kg)
gamma = 0.1  # Коэффициент затухания (1/s)
dt = 0.01  # Шаг времени (с)
T = 30.0  # Время моделирования (с)
F0 = 0.5  # Амплитуда внешней силы (N)
omega_ext = 1.5  # Частота внешней силы (rad/s)


# Функция внешней силы
def externalForce(t):
    return F0 * math.cos(omega_ext * t)


def analyticalSolutionWithForce(x0, v0):
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    omega0 = math.sqrt(k / m)
    gamma_val = gamma  # Избегаем конфликтов с именем функции

    # Определяем омега1 (допустим, без затухания)
    if omega0 > gamma_val:
        omega1 = math.sqrt(omega0 ** 2 - gamma_val ** 2)
    else:
        omega1 = 0.0  # Критическое затухание или переоснащение

    # Амплитуда и фаза частного решения
    denominator = math.sqrt((omega0 ** 2 - omega_ext ** 2) ** 2 + (gamma_val * omega_ext) ** 2)
    A = F0 / m / denominator
    phi = math.atan2(gamma_val * omega_ext, omega0 ** 2 - omega_ext ** 2)

    # Частное решение
    def x_particular(t):
        return A * math.cos(omega_ext * t - phi)

    # Производная частного решения
    def v_particular(t):
        return -A * omega_ext * math.sin(omega_ext * t - phi)

    # Коэффициенты однородного решения для удовлетворения начальных условий
    # x(t) = x_hom(t) + x_part(t)
    # x(0) = x0 = x_hom(0) + x_part(0)
    # v(0) = v0 = x'_hom(0) + x'_part(0)
    x_p0 = x_particular(0)
    v_p0 = v_particular(0)

    C = x0 - x_p0
    if omega1 != 0.0:
        D = (v0 - v_p0 + gamma_val * C) / omega1
    else:
        # В случае критического или переоснащённого затухания
        # Решение изменяется, но для простоты можно считать D=0
        D = 0.0

    print("Вычисление аналитического решения с внешней силой:")
    for i in tqdm(range(timesteps)):
        t = i * dt
        exp_term = math.exp(-gamma_val * t)
        if omega1 != 0.0:
            x_hom = exp_term * (C * math.cos(omega1 * t) + D * math.sin(omega1 * t))
        else:
            # Критическое затухание: x_hom(t) = (C + D * t) * e^(-gamma t)
            x_hom = (C + D * t) * math.exp(-gamma_val * t)
        x_p = x_particular(t)
        x_total = x_hom + x_p
        xList[i] = x_total
        timeList[i] = t

    return timeList, xList


def eulerMethodWithForce(x0, v0):
    x = x0
    v = v0
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения методом Эйлера с внешней силой:")
    for i in tqdm(range(timesteps)):
        xList[i] = x
        timeList[i] = i * dt

        t = i * dt
        a = - (k / m) * x - (gamma / m) * v + externalForce(t) / m
        x += v * dt
        v += a * dt

    return timeList, xList


def rungeKuttaWithForce(x0, v0):
    x = x0
    v = v0
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения методом Рунге-Кутты 4-го порядка с внешней силой:")
    for i in tqdm(range(timesteps)):
        xList[i] = x
        timeList[i] = i * dt

        t = i * dt

        # Промежуточные коэффициенты
        a1 = - (k / m) * x - (gamma / m) * v + externalForce(t) / m
        k1_x = dt * v
        k1_v = dt * a1

        a2 = - (k / m) * (x + 0.5 * k1_x) - (gamma / m) * (v + 0.5 * k1_v) + externalForce(t + 0.5 * dt) / m
        k2_x = dt * (v + 0.5 * k1_v)
        k2_v = dt * a2

        a3 = - (k / m) * (x + 0.5 * k2_x) - (gamma / m) * (v + 0.5 * k2_v) + externalForce(t + 0.5 * dt) / m
        k3_x = dt * (v + 0.5 * k2_v)
        k3_v = dt * a3

        a4 = - (k / m) * (x + k3_x) - (gamma / m) * (v + k3_v) + externalForce(t + dt) / m
        k4_x = dt * (v + k3_v)
        k4_v = dt * a4

        x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return timeList, xList


def plotGraph(timeEuler, xEuler, timeRK, xRK, timeAnalytical, xAnalytical):
    plt.figure(figsize=(10, 6))
    plt.title("Колебания гармонического осциллятора с внешней силой")
    plt.xlabel("Время (с)")
    plt.ylabel("Смещение (м)")

    # Метод Эйлера
    plt.plot(timeEuler, xEuler, linestyle='--', color='blue', label='Метод Эйлера')

    # Метод Рунге-Кутты
    plt.plot(timeRK, xRK, linestyle=':', color='red', label='Метод Рунге-Кутты 4-го порядка')

    # Аналитическое решение
    plt.plot(timeAnalytical, xAnalytical, linestyle='-', color='green', label='Аналитическое решение')

    plt.xlim(0, T)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def plotError(timeList, analytical, numerical, method_name):
    error = [abs(a - n) for a, n in zip(analytical, numerical)]
    plt.figure(figsize=(8, 6))
    plt.title(f"Ошибка между аналитическим и {method_name} решением")
    plt.xlabel("Время (с)")
    plt.ylabel("Абсолютная ошибка")
    plt.plot(timeList, error)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    x0 = 1.0  # Начальное смещение (м)
    v0 = 0.0  # Начальная скорость (м/с)

    # Решение методом Эйлера с внешней силой и затуханием
    timeEuler, xEuler = eulerMethodWithForce(x0, v0)

    # Решение методом Рунге-Кутты с внешней силой и затуханием
    timeRK, xRK = rungeKuttaWithForce(x0, v0)

    # Аналитическое решение с внешней силой и затуханием
    timeAnalytical, xAnalytical = analyticalSolutionWithForce(x0, v0)

    plotGraph(timeEuler, xEuler, timeRK, xRK, timeAnalytical, xAnalytical)
    #plotError(timeEuler, xAnalytical, xEuler, "методом Эйлера")
    #plotError(timeRK, xAnalytical, xRK, "методом Рунге-Кутты")

    print("Графики отображены.")


if __name__ == "__main__":
    main()
