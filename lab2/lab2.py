import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# Константы
k = 1.0      # Жёсткость пружины
m = 1.0      # Масса
gamma = 0.1  # Коэффициент затухания
dt = 0.01    # Шаг времени
T = 30.0     # Время моделирования
F0 = 0.5     # Амплитуда внешней силы
omega_ext = 1.5  # Частота внешней силы

# Функция внешней силы
def externalForce(t):
    return F0 * math.cos(omega_ext * t)

def analyticalSolution(x0, v0):
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    omega0 = math.sqrt(k / m)
    omega1 = math.sqrt(omega0 ** 2 - gamma ** 2)

    print("Вычисление аналитического решения:")
    for i in tqdm(range(timesteps)):
        t = i * dt
        exp_term = math.exp(-gamma * t)
        cos_term = math.cos(omega1 * t)
        sin_term = math.sin(omega1 * t)
        x_hom = exp_term * (x0 * cos_term + (v0 + gamma * x0) / omega1 * sin_term)
        xList[i] = x_hom
        timeList[i] = t

    return timeList, xList

def eulerMethod(x0, v0):
    x = x0
    v = v0
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения методом Эйлера:")
    for i in tqdm(range(timesteps)):
        xList[i] = x
        timeList[i] = i * dt

        a = - (k / m) * x - (gamma / m) * v
        x += v * dt
        v += a * dt

    return timeList, xList

def rungeKutta(x0, v0):
    x = x0
    v = v0
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения методом Рунге-Кутты 4-го порядка:")
    for i in tqdm(range(timesteps)):
        xList[i] = x
        timeList[i] = i * dt

        # Промежуточные коэффициенты
        a1 = - (k / m) * x - (gamma / m) * v
        k1_x = dt * v
        k1_v = dt * a1

        a2 = - (k / m) * (x + 0.5 * k1_x) - (gamma / m) * (v + 0.5 * k1_v)
        k2_x = dt * (v + 0.5 * k1_v)
        k2_v = dt * a2

        a3 = - (k / m) * (x + 0.5 * k2_x) - (gamma / m) * (v + 0.5 * k2_v)
        k3_x = dt * (v + 0.5 * k2_v)
        k3_v = dt * a3

        a4 = - (k / m) * (x + k3_x) - (gamma / m) * (v + k3_v)
        k4_x = dt * (v + k3_v)
        k4_v = dt * a4

        x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return timeList, xList

def analyticalSolutionWithForce(x0, v0):
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    omega0 = math.sqrt(k / m)
    omega1 = math.sqrt(omega0 ** 2 - gamma ** 2)

    print("Вычисление аналитического решения:")

    # Амплитуда и фаза частного решения
    denominator = ((omega0 ** 2 - omega_ext ** 2) ** 2 + (gamma * omega_ext) ** 2)
    A = F0 / m / math.sqrt(denominator)
    phi = math.atan2(gamma * omega_ext, omega0 ** 2 - omega_ext ** 2)

    for i in tqdm(range(timesteps)):
        t = i * dt
        exp_term = math.exp(-gamma * t)
        cos_term = math.cos(omega1 * t)
        sin_term = math.sin(omega1 * t)
        x_hom = exp_term * (x0 * cos_term + (v0 + gamma * x0) / omega1 * sin_term)
        x_part = A * math.cos(omega_ext * t - phi)
        xList[i] = x_hom + x_part
        timeList[i] = t

    return timeList, xList

def eulerMethodWithForce(x0, v0):
    x = x0
    v = v0
    timesteps = int(T / dt)
    xList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения методом Эйлера:")
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

    print("Вычисление численного решения методом Рунге-Кутты 4-го порядка:")
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
    plt.title("Колебания гармонического осциллятора")
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
    plt.savefig(f'Ошибка_{method_name}.png')
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
    plotError(timeEuler, xAnalytical, xEuler, "методом Эйлера")
    plotError(timeRK, xAnalytical, xRK, "методом Рунге-Кутты")

    print("График сохранён как 'oscillator.png'")

if __name__ == "__main__":
    main()
