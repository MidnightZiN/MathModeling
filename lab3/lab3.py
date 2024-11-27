import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры
g = 9.81  # Ускорение свободного падения, м/с^2


def compute_force(x, y, L, k):
    """
    Вычисляет силы и ускорения с учетом растяжения нити.
    """
    r = np.sqrt(x ** 2 + y ** 2)  # Расстояние от точки подвеса
    if r > L:  # Если нить растянута
        tension_force = k * (r - L)  # Пропорционально растягиванию
        ax = -tension_force * x / r
        ay = -tension_force * y / r - g  # Учитываем гравитацию
    else:  # Если нить не растянута
        ax = 0
        ay = -g  # Только гравитация
    return ax, ay


def simulate_pendulum_bouncing(x0, y0, vx0, vy0, L, k, T, dt):
    """
    Симуляция движения маятника с отскоками.
    """
    times = np.arange(0, T, dt)
    x, y = np.zeros(len(times)), np.zeros(len(times))
    vx, vy = np.zeros(len(times)), np.zeros(len(times))
    x[0], y[0], vx[0], vy[0] = x0, y0, vx0, vy0

    for i in range(1, len(times)):
        ax, ay = compute_force(x[i - 1], y[i - 1], L, k)
        vx[i] = vx[i - 1] + ax * dt
        vy[i] = vy[i - 1] + ay * dt
        x[i] = x[i - 1] + vx[i] * dt
        y[i] = y[i - 1] + vy[i] * dt

        # Если маятник достигает максимального растяжения нити
        r = np.sqrt(x[i] ** 2 + y[i] ** 2)
        if r > L:
            # Ограничиваем длину нити
            x[i] = x[i] * L / r
            y[i] = y[i] * L / r
            # Вычисляем радиальную скорость и инвертируем её для имитации отскока
            v_radial = (vx[i] * x[i] + vy[i] * y[i]) / L  # Составляющая вдоль радиуса
            vx[i] -= 2 * v_radial * x[i] / L
            vy[i] -= 2 * v_radial * y[i] / L

    return times, x, y


def visualize_pendulum_bouncing(times, x, y, L):
    """
    Визуализация движения маятника с отскоками.
    Накладывает траекторию на анимацию маятника.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-L - 1, L + 1)
    ax.set_ylim(-L - 1, L + 1)
    ax.set_aspect('equal')
    ax.set_title("Маятник с отскоками и его траектория")
    ax.set_xlabel("Координата x")
    ax.set_ylabel("Координата y")

    # Объекты для анимации маятника и траектории
    line, = ax.plot([], [], 'o-', lw=2, label='Маятник')
    trajectory, = ax.plot([], [], 'r-', lw=1.5, label='Траектория')

    # Легенда
    ax.legend()

    def update(frame):
        # Обновляем положение маятника
        line.set_data([0, x[frame]], [0, y[frame]])
        # Обновляем траекторию
        trajectory.set_data(x[:frame + 1], y[:frame + 1])
        return line, trajectory

    ani = FuncAnimation(fig, update, frames=len(times), interval=20, blit=True)
    plt.show()


# Основная программа
if __name__ == "__main__":
    # Начальные параметры
    L = 1.1  # Длина нити
    x0, y0 = -0.3, 0.6  # Начальная координата
    vx0, vy0 = 0.0, 2.8  # Начальная скорость
    k = 8.0  # Коэффициент упругости
    T = 30.0  # Время моделирования
    dt = 0.01  # Шаг времени

    # Симуляция
    times, x, y = simulate_pendulum_bouncing(x0, y0, vx0, vy0, L, k, T, dt)

    # Визуализация
    visualize_pendulum_bouncing(times, x, y, L)
