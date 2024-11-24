import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Константы
g = 9.81  # Ускорение свободного падения
L0 = 5.0  # Начальная длина маятника
dt = 0.05  # Шаг времени
T = 20.0  # Время моделирования
k = 1  # Коэффициент ослабления нити (скорость увеличения длины)

def eulerVariableLength(alpha0, omega0):
    alpha = alpha0
    omega = omega0
    L = L0
    timesteps = int(T / dt)
    alphaList = [0.0] * timesteps
    lengthList = [0.0] * timesteps

    print("Вычисление численного решения с изменяющейся длиной нити:")
    for i in tqdm(range(timesteps)):
        alphaList[i] = alpha
        lengthList[i] = L

        # Обновление длины нити
        L += k * dt  # Нить ослабляется и длина увеличивается

        # Угловое ускорение с учётом текущей длины
        omega_dot = - (g / L) * math.sin(alpha)
        omega += omega_dot * dt
        alpha += omega * dt

    return alphaList, lengthList

def animatePendulum(alphaList, lengthList, save_path='variable_pendulum_animation.gif'):
    # Определяем максимальную длину нити для корректного масштабирования
    max_length = max(lengthList)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-max_length - 1, max_length + 1)  # Горизонтальные границы
    ax.set_ylim(-max_length - 1, max_length * 0.2)  # Верхняя граница выше, нижняя увеличена
    ax.set_aspect('equal', 'box')
    ax.grid()

    # Линия и точка для маятника
    line, = ax.plot([], [], 'o-', lw=2, color='blue')
    time_text = ax.text(-max_length, max_length * 0.15, '', fontsize=10, color='red')
    length_text = ax.text(-max_length, max_length * 0.1, '', fontsize=10, color='green')

    def init():
        line.set_data([], [])
        time_text.set_text('')
        length_text.set_text('')
        return line, time_text, length_text

    def update(frame):
        alpha = alphaList[frame]
        L = lengthList[frame]

        # Координаты маятника
        x = L * math.sin(alpha)
        y = -L * math.cos(alpha) + max_length * 0.01  # Смещение подвеса выше

        line.set_data([0, x], [max_length * 0.01, y])

        # Обновление текста
        time_text.set_text(f'Time: {frame * dt:.2f} s')
        length_text.set_text(f'Length: {L:.2f} m')
        return line, time_text, length_text

    # Ускорение воспроизведения через fps и interval
    anim = FuncAnimation(fig, update, frames=len(alphaList), init_func=init, blit=True, interval=dt * 500)
    anim.save(save_path, fps=15, writer='pillow')
    plt.close(fig)

def main():
    alpha0 = 0.5  # Начальный угол (в радианах)
    omega0 = 0.0  # Начальная угловая скорость

    # Решение методом Эйлера с изменяющейся длиной нити
    alphaList, lengthList = eulerVariableLength(alpha0, omega0)

    # Анимация маятника
    animatePendulum(alphaList, lengthList)
    print("Анимация сохранена как 'variable_pendulum_animation.gif'")

if __name__ == "__main__":
    main()
