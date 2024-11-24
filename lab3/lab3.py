import math
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import numpy as np

# Константы
g = 9.81  # Ускорение свободного падения (м/с^2)
L = 1.0  # Длина нити маятника (м)
dt = 0.001  # Шаг времени (с)
T = 10.0  # Время моделирования (с)

# Начальные условия
theta0 = math.radians(120)  # Начальный угол отклонения (рад)
omega0 = 0.0  # Начальная угловая скорость (рад/с)


# Функция для расчета натяжения нити
def tension(theta, omega):
    Tension = L * omega ** 2 + g * math.cos(theta)
    return Tension


def simulate(T, dt, theta0, omega0):
    timesteps = int(T / dt)
    timeList = np.linspace(0, T, timesteps)

    # Списки для хранения результатов
    thetaList = []
    omegaList = []
    xList = []
    yList = []

    # Начальные условия
    theta = theta0
    omega = omega0
    x = L * math.sin(theta)
    y = -L * math.cos(theta)
    vx = 0.0
    vy = 0.0

    mode = 'swing'  # Режим: 'swing' (маятник) или 'freefall' (свободное падение)

    for t in timeList:
        if mode == 'swing':
            # Вычисляем натяжение нити
            Tension = tension(theta, omega)
            if Tension > 0:
                # Уравнения движения для маятника
                alpha = -(g / L) * math.sin(theta)
                omega += alpha * dt
                theta += omega * dt
                x = L * math.sin(theta)
                y = -L * math.cos(theta)
                vx = L * omega * math.cos(theta)
                vy = L * omega * math.sin(theta)
            else:
                # Нить ослабла, переходим в режим свободного падения
                mode = 'freefall'
                # Преобразуем полярные координаты в декартовы скорости
                vx = L * omega * math.cos(theta)
                vy = L * omega * math.sin(theta)
        elif mode == 'freefall':
            # Уравнения движения для свободного падения
            vx = vx
            vy = vy + (-g) * dt
            x += vx * dt
            y += vy * dt
            # Проверяем, натянулась ли нить
            r = math.sqrt(x ** 2 + y ** 2)
            if r >= L:
                mode = 'swing'
                theta = math.atan2(x, -y)
                omega = (x * vy - y * vx) / (L ** 2)
                # Проецируем x и y на окружность радиуса L
                x = L * math.sin(theta)
                y = -L * math.cos(theta)
                vx = L * omega * math.cos(theta)
                vy = L * omega * math.sin(theta)

        # Сохраняем результаты
        thetaList.append(theta)
        omegaList.append(omega)
        xList.append(x)
        yList.append(y)

    return timeList, xList, yList, thetaList, omegaList


def animatePendulum(xList, yList, dt):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)  # Оставляем место для слайдера
    ax.set_xlim(-1.2 * L, 1.2 * L)
    ax.set_ylim(-1.2 * L, 1.2 * L)
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '-', lw=1, color='gray', alpha=0.5)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    xdata, ydata = [], []

    # Добавляем ось для слайдера скорости
    axspeed = plt.axes([0.25, 0.1, 0.65, 0.03])
    speed_slider = Slider(
        ax=axspeed,
        label='Скорость',
        valmin=1.0,
        valmax=50.0,
        valinit=1.0,
    )

    # Функция инициализации
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return line, trace, time_text

    # Индекс текущего кадра
    frame_idx = [0]
    total_frames = len(xList)

    # Функция обновления кадров
    def update(frame):
        idx = frame_idx[0]
        x = [0, xList[idx]]
        y = [0, yList[idx]]
        line.set_data(x, y)
        xdata.append(xList[idx])
        ydata.append(yList[idx])
        trace.set_data(xdata, ydata)
        time_text.set_text(f'Время = {idx * dt:.2f} с')

        # Обновляем индекс кадра в соответствии со скоростью
        frame_idx[0] += int(speed_slider.val)
        if frame_idx[0] >= total_frames:
            frame_idx[0] = 0
            xdata.clear()
            ydata.clear()
        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, update, init_func=init, blit=True, interval=20
    )

    plt.show()


def main():
    # Вызываем функцию simulate с необходимыми параметрами
    timeList, xList, yList, thetaList, omegaList = simulate(T, dt, theta0, omega0)
    # plotTrajectory(xList, yList)  # Можно отключить для быстрого запуска анимации
    animatePendulum(xList, yList, dt)


if __name__ == "__main__":
    main()
