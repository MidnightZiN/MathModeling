import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from numba import njit

# Константы
g = 9.81  # Ускорение свободного падения (м/с^2)
dt = 0.01  # Шаг времени (с)
T = 10.0  # Общее время моделирования (с)
e = 0.9  # Коэффициент восстановления при столкновении (0 < e < 1)
x_min, x_max = 0, 10  # Границы по оси X (м)
y_min, y_max = 0, 10  # Границы по оси Y (м)
r = 0.2  # Радиус шарика (м)


@njit
def eulerMethod(x0, y0, vx0, vy0):
    x = x0
    y = y0
    vx = vx0
    vy = vy0
    timesteps = int(T / dt)
    xList = np.zeros(timesteps)
    yList = np.zeros(timesteps)
    timeList = np.zeros(timesteps)

    for i in range(timesteps):
        xList[i] = x
        yList[i] = y
        timeList[i] = i * dt

        # Обновление скоростей с учетом ускорения свободного падения
        vy -= g * dt

        # Обновление положений
        x += vx * dt
        y += vy * dt

        # Проверка столкновений со стенками
        if x - r <= x_min:
            x = x_min + r
            vx = -vx * e
        if x + r >= x_max:
            x = x_max - r
            vx = -vx * e
        if y - r <= y_min:
            y = y_min + r
            vy = -vy * e
        if y + r >= y_max:
            y = y_max - r
            vy = -vy * e

    return timeList, xList, yList


@njit
def rungeKuttaMethod(x0, y0, vx0, vy0):
    x = x0
    y = y0
    vx = vx0
    vy = vy0
    timesteps = int(T / dt)
    xList = np.zeros(timesteps)
    yList = np.zeros(timesteps)
    timeList = np.zeros(timesteps)

    for i in range(timesteps):
        xList[i] = x
        yList[i] = y
        timeList[i] = i * dt

        # Определение функций для ускорений
        def ax(t, x, y, vx, vy):
            return 0  # Нет ускорения по оси X

        def ay(t, x, y, vx, vy):
            return -g  # Ускорение свободного падения по оси Y

        # Коэффициенты Рунге-Кутты
        k1_vx = dt * ax(timeList[i], x, y, vx, vy)
        k1_vy = dt * ay(timeList[i], x, y, vx, vy)
        k1_x = dt * vx
        k1_y = dt * vy

        k2_vx = dt * ax(timeList[i] + 0.5 * dt, x + 0.5 * k1_x, y + 0.5 * k1_y, vx + 0.5 * k1_vx, vy + 0.5 * k1_vy)
        k2_vy = dt * ay(timeList[i] + 0.5 * dt, x + 0.5 * k1_x, y + 0.5 * k1_y, vx + 0.5 * k1_vx, vy + 0.5 * k1_vy)
        k2_x = dt * (vx + 0.5 * k1_vx)
        k2_y = dt * (vy + 0.5 * k1_vy)

        k3_vx = dt * ax(timeList[i] + 0.5 * dt, x + 0.5 * k2_x, y + 0.5 * k2_y, vx + 0.5 * k2_vx, vy + 0.5 * k2_vy)
        k3_vy = dt * ay(timeList[i] + 0.5 * dt, x + 0.5 * k2_x, y + 0.5 * k2_y, vx + 0.5 * k2_vx, vy + 0.5 * k2_vy)
        k3_x = dt * (vx + 0.5 * k2_vx)
        k3_y = dt * (vy + 0.5 * k2_vy)

        k4_vx = dt * ax(timeList[i] + dt, x + k3_x, y + k3_y, vx + k3_vx, vy + k3_vy)
        k4_vy = dt * ay(timeList[i] + dt, x + k3_x, y + k3_y, vx + k3_vx, vy + k3_vy)
        k4_x = dt * (vx + k3_vx)
        k4_y = dt * (vy + k3_vy)

        # Обновление скоростей и положений
        vx += (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6
        vy += (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
        x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6

        # Проверка столкновений со стенками
        if x - r <= x_min:
            x = x_min + r
            vx = -vx * e
        if x + r >= x_max:
            x = x_max - r
            vx = -vx * e
        if y - r <= y_min:
            y = y_min + r
            vy = -vy * e
        if y + r >= y_max:
            y = y_max - r
            vy = -vy * e

    return timeList, xList, yList


def plotTrajectory(timeEuler, xEuler, yEuler, timeRK, xRK, yRK):
    plt.figure(figsize=(10, 6))
    plt.title("Траектория движения шарика")
    plt.plot(xEuler, yEuler, linestyle='--', color='blue', label='Метод Эйлера')
    plt.plot(xRK, yRK, linestyle='-', color='red', label='Метод Рунге-Кутты 4-го порядка')
    plt.xlabel("X (м)")
    plt.ylabel("Y (м)")
    plt.legend()
    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig('ball_trajectory.png')
    plt.close()


def plotPosition(timeEuler, xEuler, yEuler, timeRK, xRK, yRK):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("Положение шарика по оси X")
    plt.plot(timeEuler, xEuler, linestyle='--', color='blue', label='Метод Эйлера')
    plt.plot(timeRK, xRK, linestyle='-', color='red', label='Метод Рунге-Кутты 4-го порядка')
    plt.ylabel("X (м)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title("Положение шарика по оси Y")
    plt.plot(timeEuler, yEuler, linestyle='--', color='green', label='Метод Эйлера')
    plt.plot(timeRK, yRK, linestyle='-', color='purple', label='Метод Рунге-Кутты 4-го порядка')
    plt.xlabel("Время (с)")
    plt.ylabel("Y (м)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ball_position.png')
    plt.close()


def createAnimation(xRK, yRK):
    # Пропускаем кадры для уменьшения общего количества
    frame_step = 10  # Используем каждый 10-й кадр
    xRK = xRK[::frame_step]
    yRK = yRK[::frame_step]
    total_frames = len(xRK)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_title('Анимация движения шарика')
    ax.grid(True)

    circle = plt.Circle((xRK[0], yRK[0]), r, color='red')
    ax.add_patch(circle)

    def init():
        circle.center = (xRK[0], yRK[0])
        return circle,

    def update(frame):
        circle.center = (xRK[frame], yRK[frame])
        return circle,

    print("Создание анимации:")
    ani = FuncAnimation(fig, update, frames=range(total_frames), init_func=init, blit=True, interval=10)

    # Добавляем прогресс бар при сохранении анимации
    print("Сохранение анимации в файл GIF:")
    with tqdm(total=total_frames) as pbar:
        def update_progress(current_frame, total_frames):
            pbar.update(1)

        ani.save('ball_bouncing_animation.gif', writer=PillowWriter(fps=60), progress_callback=update_progress)

    plt.close()


def main():
    # Начальные условия
    x0 = 1.0  # Начальное положение по X (м)
    y0 = 5.0  # Начальное положение по Y (м)
    vx0 = 2.0  # Начальная скорость по X (м/с)
    vy0 = 0.0  # Начальная скорость по Y (м/с)

    print("Вычисление численного решения методом Эйлера:")
    timeEuler, xEuler, yEuler = eulerMethod(x0, y0, vx0, vy0)
    print("Метод Эйлера завершён.")

    print("Вычисление численного решения методом Рунге-Кутты 4-го порядка:")
    timeRK, xRK, yRK = rungeKuttaMethod(x0, y0, vx0, vy0)
    print("Метод Рунге-Кутты завершён.")

    # Построение графиков
    plotTrajectory(timeEuler, xEuler, yEuler, timeRK, xRK, yRK)
    plotPosition(timeEuler, xEuler, yEuler, timeRK, xRK, yRK)
    print("Графики сохранены как 'ball_trajectory.png' и 'ball_position.png'.")

    # Создание анимации
    createAnimation(xRK, yRK)
    print("Анимация сохранена как 'ball_bouncing_animation.gif'")


if __name__ == "__main__":
    main()

#Ступенчатая функция