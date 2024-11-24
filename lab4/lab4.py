import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from numba import njit

# Константы
g = 9.81  # Ускорение свободного падения (м/с^2)
m = 1.0  # Масса маятника (кг)
k = 10.0  # Жесткость пружины (Н/м)
L0 = 1.0  # Начальная длина пружины (м)
dt = 0.001  # Шаг времени (с)
T = 10.0  # Общее время моделирования (с)


@njit
def eulerMethod(theta0, omega0, r0, vr0):
    theta = theta0
    omega = omega0
    r = r0
    vr = vr0
    timesteps = int(T / dt)
    thetaList = np.zeros(timesteps)
    rList = np.zeros(timesteps)
    timeList = np.zeros(timesteps)

    for i in range(timesteps):
        thetaList[i] = theta
        rList[i] = r
        timeList[i] = i * dt

        # Вычисление ускорений
        a_theta = - (g * math.sin(theta) + 2 * vr * omega) / r
        a_r = r * omega ** 2 - (k / m) * (r - L0) - g * math.cos(theta)

        # Обновление скоростей
        omega += a_theta * dt
        vr += a_r * dt

        # Обновление положений
        theta += omega * dt
        r += vr * dt

    return timeList, thetaList, rList


@njit
def rungeKuttaMethod(theta0, omega0, r0, vr0):
    theta = theta0
    omega = omega0
    r = r0
    vr = vr0
    timesteps = int(T / dt)
    thetaList = np.zeros(timesteps)
    rList = np.zeros(timesteps)
    timeList = np.zeros(timesteps)

    for i in range(timesteps):
        thetaList[i] = theta
        rList[i] = r
        timeList[i] = i * dt

        # Определение функций для ускорений
        def a_theta(theta, omega, r, vr):
            return - (g * math.sin(theta) + 2 * vr * omega) / r

        def a_r(theta, omega, r):
            return r * omega ** 2 - (k / m) * (r - L0) - g * math.cos(theta)

        # Коэффициенты для theta и omega
        k1_theta = dt * omega
        k1_omega = dt * a_theta(theta, omega, r, vr)
        k1_r = dt * vr
        k1_vr = dt * a_r(theta, omega, r)

        k2_theta = dt * (omega + 0.5 * k1_omega)
        k2_omega = dt * a_theta(theta + 0.5 * k1_theta, omega + 0.5 * k1_omega, r + 0.5 * k1_r, vr + 0.5 * k1_vr)
        k2_r = dt * (vr + 0.5 * k1_vr)
        k2_vr = dt * a_r(theta + 0.5 * k1_theta, omega + 0.5 * k1_omega, r + 0.5 * k1_r)

        k3_theta = dt * (omega + 0.5 * k2_omega)
        k3_omega = dt * a_theta(theta + 0.5 * k2_theta, omega + 0.5 * k2_omega, r + 0.5 * k2_r, vr + 0.5 * k2_vr)
        k3_r = dt * (vr + 0.5 * k2_vr)
        k3_vr = dt * a_r(theta + 0.5 * k2_theta, omega + 0.5 * k2_omega, r + 0.5 * k2_r)

        k4_theta = dt * (omega + k3_omega)
        k4_omega = dt * a_theta(theta + k3_theta, omega + k3_omega, r + k3_r, vr + k3_vr)
        k4_r = dt * (vr + k3_vr)
        k4_vr = dt * a_r(theta + k3_theta, omega + k3_omega, r + k3_r)

        # Обновление значений
        theta += (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6
        omega += (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6
        r += (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
        vr += (k1_vr + 2 * k2_vr + 2 * k3_vr + k4_vr) / 6

    return timeList, thetaList, rList


def plotResults(timeEuler, thetaEuler, rEuler, timeRK, thetaRK, rRK):
    plt.figure(figsize=(12, 8))

    # Угловое отклонение
    plt.subplot(2, 1, 1)
    plt.title("Маятник на пружине: угловое отклонение")
    plt.plot(timeEuler, thetaEuler, linestyle='--', color='blue', label='Метод Эйлера')
    plt.plot(timeRK, thetaRK, linestyle='-', color='red', label='Метод Рунге-Кутты 4-го порядка')
    plt.ylabel("Угол (рад)")
    plt.legend()
    plt.grid(True)

    # Радиальное смещение
    plt.subplot(2, 1, 2)
    plt.title("Маятник на пружине: радиальное смещение")
    plt.plot(timeEuler, rEuler, linestyle='--', color='green', label='Метод Эйлера')
    plt.plot(timeRK, rRK, linestyle='-', color='purple', label='Метод Рунге-Кутты 4-го порядка')
    plt.xlabel("Время (с)")
    plt.ylabel("Радиальное смещение (м)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('spring_pendulum_comparison.png')
    plt.close()


def plotTrajectory(thetaRK, rRK):
    xRK = rRK * np.sin(thetaRK)
    yRK = -rRK * np.cos(thetaRK)

    plt.figure(figsize=(6, 6))
    plt.plot(xRK, yRK, color='red')
    plt.title("Траектория движения маятника на пружине (Метод Рунге-Кутты)")
    plt.xlabel("X (м)")
    plt.ylabel("Y (м)")
    plt.grid(True)
    plt.savefig('spring_pendulum_trajectory.png')
    plt.close()


def createAnimation(thetaRK, rRK):
    # Пропускаем кадры для уменьшения общего количества
    frame_step = 10  # Используем каждый 10-й кадр
    xRK = rRK * np.sin(thetaRK)
    yRK = -rRK * np.cos(thetaRK)

    xRK = xRK[::frame_step]
    yRK = yRK[::frame_step]
    total_frames = len(xRK)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_title('Анимация движения маятника на пружине')
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        x = [0, xRK[frame]]
        y = [0, yRK[frame]]
        line.set_data(x, y)
        return line,

    print("Создание анимации:")
    ani = FuncAnimation(fig, update, frames=range(total_frames), init_func=init, blit=True, interval=10)

    # Добавляем прогресс бар при сохранении анимации
    print("Сохранение анимации в файл GIF:")
    with tqdm(total=total_frames) as pbar:
        def update_progress(current_frame, total_frames):
            pbar.update(1)

        ani.save('spring_pendulum_animation.gif', writer=PillowWriter(fps=60), progress_callback=update_progress)

    plt.close()


def main():
    # Начальные условия
    theta0 = 0.5  # Начальный угол (рад)
    omega0 = 0.0  # Начальная угловая скорость (рад/с)
    r0 = L0  # Начальная длина пружины (м)
    vr0 = 0.0  # Начальная радиальная скорость (м/с)

    print("Вычисление численного решения методом Эйлера:")
    timeEuler, thetaEuler, rEuler = eulerMethod(theta0, omega0, r0, vr0)
    print("Метод Эйлера завершён.")

    print("Вычисление численного решения методом Рунге-Кутты 4-го порядка:")
    timeRK, thetaRK, rRK = rungeKuttaMethod(theta0, omega0, r0, vr0)
    print("Метод Рунге-Кутты завершён.")

    # Построение графиков
    plotResults(timeEuler, thetaEuler, rEuler, timeRK, thetaRK, rRK)
    plotTrajectory(thetaRK, rRK)
    print("Графики сохранены как 'spring_pendulum_comparison.png' и 'spring_pendulum_trajectory.png'.")

    # Создание анимации
    createAnimation(thetaRK, rRK)
    print("Анимация сохранена как 'spring_pendulum_animation.gif'")


if __name__ == "__main__":
    main()
