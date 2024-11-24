import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# Константы
g = 9.81  # Ускорение свободного падения
L = 5.0   # Длина маятника
dt = 0.01 # Шаг времени
T = 10.0  # Время моделирования

def analyticalSolution(alpha0, omega0):
    timesteps = int(T / dt)
    alphaList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    omega = math.sqrt(g / L)

    print("Вычисление аналитического решения:")
    for i in tqdm(range(timesteps)):
        t = i * dt
        alphaList[i] = alpha0 * math.cos(omega * t) + (omega0 / omega) * math.sin(omega * t)
        timeList[i] = t

    return timeList, alphaList

# Решение с использованием sin(α) ≈ α
def eulerLinear(alpha0, omega0):
    alpha = alpha0
    omega = omega0
    timesteps = int(T / dt)
    alphaList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения с линейной аппроксимацией:")
    for i in tqdm(range(timesteps)):
        alphaList[i] = alpha
        timeList[i] = i * dt

        omega += -g / L * alpha * dt
        alpha += omega * dt

    return timeList, alphaList

# Решение с использованием sin(α)
def eulerExact(alpha0, omega0):
    alpha = alpha0
    omega = omega0
    timesteps = int(T / dt)
    alphaList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения с точной функцией синуса:")
    for i in tqdm(range(timesteps)):
        alphaList[i] = alpha
        timeList[i] = i * dt

        omega += -g / L * math.sin(alpha) * dt
        alpha += omega * dt

    return timeList, alphaList

def plotGraph(timeLinear, alphaLinear, timeExact, alphaExact, timeAnalytical, alphaAnalytical, timeRK, alphaRK):
    plt.figure(figsize=(10, 6))
    plt.title("Колебания маятника")
    plt.xlabel("Время (с)")
    plt.ylabel("Угол отклонения (рад)")

    # Линейное приближение (Эйлер)
    plt.plot(timeLinear, alphaLinear, linestyle='--', color='blue', label='Эйлер с линейной аппроксимацией')

    # Точное численное решение (Эйлер)
    plt.plot(timeExact, alphaExact, linestyle=':', color='red', label='Эйлер с точной функцией синуса')

    # Аналитическое решение
    plt.plot(timeAnalytical, alphaAnalytical, linestyle='-', color='green', label='Аналитическое решение')

    # Решение методом Рунге-Кутты
    #plt.plot(timeRK, alphaRK, linestyle='-.', color='purple', label='Рунге-Кутта 4-го порядка')

    plt.xlim(0, T)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pendulum.png')
    plt.close()

# Решение методом Рунге-Кутты 4-го порядка
def rungeKutta(alpha0, omega0):
    alpha = alpha0
    omega = omega0
    timesteps = int(T / dt)
    alphaList = [0.0] * timesteps
    timeList = [0.0] * timesteps

    print("Вычисление численного решения методом Рунге-Кутты 4-го порядка:")
    for i in tqdm(range(timesteps)):
        timeList[i] = i * dt
        alphaList[i] = alpha

        # Промежуточные коэффициенты для alpha и omega
        k1_alpha = dt * omega
        k1_omega = dt * (-g / L * math.sin(alpha))

        k2_alpha = dt * (omega + 0.5 * k1_omega)
        k2_omega = dt * (-g / L * math.sin(alpha + 0.5 * k1_alpha))

        k3_alpha = dt * (omega + 0.5 * k2_omega)
        k3_omega = dt * (-g / L * math.sin(alpha + 0.5 * k2_alpha))

        k4_alpha = dt * (omega + k3_omega)
        k4_omega = dt * (-g / L * math.sin(alpha + k3_alpha))

        # Обновление значений alpha и omega
        alpha += (k1_alpha + 2 * k2_alpha + 2 * k3_alpha + k4_alpha) / 6
        omega += (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6

    return timeList, alphaList


def plotError(timeList, analytical, numerical, method_name):
    error = [abs(a - n) for a, n in zip(analytical, numerical)]
    plt.figure(figsize=(8, 6))
    plt.title(f"Ошибка между аналитическим и {method_name} решением")
    plt.xlabel("Время (с)")
    plt.ylabel("Абсолютная ошибка")
    plt.plot(timeList, error)
    plt.savefig(f'Отличия_{method_name}.png')
    plt.close()

def main():
    alpha0 = 1.0  # Начальный угол (в радианах)
    omega0 = 0.5  # Начальная угловая скорость

    # Приближенное решение
    timeLinear, alphaLinear = eulerLinear(alpha0, omega0)

    # Точное численное решение
    timeExact, alphaExact = eulerExact(alpha0, omega0)

    # Аналитическое решение
    timeAnalytical, alphaAnalytical = analyticalSolution(alpha0, omega0)

    # Решение методом Рунге-Кутты 4-го порядка
    timeRK, alphaRK = rungeKutta(alpha0, omega0)

    plotGraph(timeLinear, alphaLinear, timeExact, alphaExact, timeAnalytical, alphaAnalytical, timeRK, alphaRK)
    plotError(timeLinear, alphaAnalytical, alphaLinear, "Эйлер с аппроксимацией")
    plotError(timeLinear, alphaAnalytical, alphaExact, "Эйлер с точным")

    print("График сохранён как 'pendulum.png'")

if __name__ == "__main__":
    main()
