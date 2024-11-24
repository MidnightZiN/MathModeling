import math
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import numpy as np

# Физические константы и начальные параметры
G = 9.8  # Ускорение свободного падения (м/с^2)
DT = 0.01  # Шаг времени для симуляции (с)
BALL_RADIUS = 3  # Радиус шарика
X_MIN, X_MAX = -120, 120  # Границы по оси X

# Параметры синусоидальных границ
SIN_PARAMS = {
    "lower": {"A": 15, "omega": 0.1, "phi": 15, "D": -50},
    "upper": {"A": 15, "omega": 0.3, "phi": 25, "D": 50}
}

# Начальные координаты и скорости мяча
INIT_PARAMS = {"x": 0, "y": 0, "U": 20, "V": -20}


def calculate_sin(x, A, omega, phi, D):
    """Вычисляет значение синусоиды в точке x."""
    return A * math.sin(omega * x + phi) + D


def calculate_normal(x, A, omega, phi, D):
    """Вычисляет нормаль к синусоиде в точке x."""
    dy_dx = A * omega * math.cos(omega * x + phi)
    norm = math.sqrt(dy_dx ** 2 + 1)
    return -dy_dx / norm, 1 / norm


def update_velocity(U, V, nx, ny):
    """Обновляет скорость после столкновения с поверхностью."""
    dot_product = U * nx + V * ny
    U_new = U - 2 * dot_product * nx
    V_new = V - 2 * dot_product * ny
    return U_new, V_new


def simulate_motion(x0, y0, U0, V0, DT, total_time):
    """Симулирует движение шарика и возвращает списки координат x и y."""
    x_list = []
    y_list = []
    x, y, U, V = x0, y0, U0, V0
    num_steps = int(total_time / DT)

    for _ in range(num_steps):
        x_new = x + U * DT
        y_new = y + V * DT - 0.5 * G * DT ** 2
        V -= G * DT

        # Проверка столкновений с вертикальными границами
        if x_new <= X_MIN or x_new >= X_MAX:
            U = -U

        # Проверка столкновений с синусоидальными границами
        for boundary, params in SIN_PARAMS.items():
            A, omega, phi, D = params.values()
            sin_y = calculate_sin(x_new, A, omega, phi, D)
            if (boundary == "lower" and y_new <= sin_y) or (boundary == "upper" and y_new >= sin_y):
                nx, ny = calculate_normal(x_new, A, omega, phi, D)
                U, V = update_velocity(U, V, nx, ny)
                y_new = sin_y  # Корректируем позицию по границе
                break

        x, y = x_new, y_new
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


class AnimationHandler:
    def __init__(self):
        # Инициализация фигуры и осей
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)  # Оставляем место для слайдера
        self.ball = plt.Circle((INIT_PARAMS["x"], INIT_PARAMS["y"]), BALL_RADIUS, color='green')
        self.init_graphics()

        # Симуляция движения и сохранение позиций
        total_time = 30  # Общее время симуляции в секундах
        self.x_positions, self.y_positions = simulate_motion(
            INIT_PARAMS["x"], INIT_PARAMS["y"], INIT_PARAMS["U"], INIT_PARAMS["V"], DT, total_time
        )

        self.num_frames = len(self.x_positions)
        self.frame_idx = 0
        self.frame_accum = 0.0

        # Добавляем слайдер скорости
        self.speed_slider = self.add_speed_slider()

    def init_graphics(self):
        """Инициализация графики."""
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor("white")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(X_MIN, X_MAX)

        # Вычисляем пределы по оси Y на основе синусоид
        x_values = np.linspace(X_MIN, X_MAX, 1000)
        y_min = np.inf
        y_max = -np.inf
        for params in SIN_PARAMS.values():
            A, omega, phi, D = params.values()
            y_values = A * np.sin(omega * x_values + phi) + D
            y_min = min(y_min, np.min(y_values))
            y_max = max(y_max, np.max(y_values))
            self.ax.plot(x_values, y_values, color="blue")

        y_range = y_max - y_min
        y_margin = 0.1 * y_range
        y_min_plot = y_min - y_margin
        y_max_plot = y_max + y_margin
        self.ax.set_ylim(y_min_plot, y_max_plot)

        # Вертикальные границы
        self.ax.axvline(X_MIN, color="red")
        self.ax.axvline(X_MAX, color="red")

        # Добавляем шарик
        self.ax.add_patch(self.ball)

    def add_speed_slider(self):
        """Добавляет слайдер скорости на график."""
        axspeed = plt.axes([0.25, 0.1, 0.5, 0.03])
        speed_slider = Slider(
            ax=axspeed,
            label='Скорость',
            valmin=0.0,
            valmax=50.0,
            valinit=0.0,  # Начальное значение - остановка времени
        )
        return speed_slider

    def update(self, frame):
        """Функция обновления для анимации."""
        # Получаем скорость из слайдера
        speed = self.speed_slider.val
        # Накопление индекса кадра
        self.frame_accum += speed
        if self.frame_accum >= 1.0:
            increment = int(self.frame_accum)
            self.frame_accum -= increment
            self.frame_idx = (self.frame_idx + increment) % self.num_frames
        # Обновляем позицию шарика
        x = self.x_positions[self.frame_idx]
        y = self.y_positions[self.frame_idx]
        self.ball.set_center((x, y))
        return [self.ball]

    def run(self):
        """Запуск анимации."""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.num_frames,
            interval=20,
            blit=True
        )
        plt.show()


if __name__ == '__main__':
    AnimationHandler().run()
