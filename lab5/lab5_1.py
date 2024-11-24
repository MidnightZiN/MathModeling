import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Константы физики и начальные параметры
G = 9.8
HT = 0.1
BALL_RADIUS = 1
X_MIN, X_MAX = -120, 120

# Параметры синусоид
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
    """Обновляет скорость после столкновения."""
    dot_product = U * nx + V * ny
    return U - 2 * dot_product * nx, V - 2 * dot_product * ny


def get_pos(x0, y0, U0, V0):
    """Генератор позиций мяча."""
    x, y, U, V = x0, y0, U0, V0

    while True:
        x_new = x + U * HT
        y_new = y + V * HT - 0.5 * G * HT ** 2
        V -= G * HT

        # Проверка столкновений
        if x_new <= X_MIN or x_new >= X_MAX:
            U = -U

        for boundary, params in SIN_PARAMS.items():
            A, omega, phi, D = params.values()
            sin_y = calculate_sin(x_new, A, omega, phi, D)
            if (boundary == "lower" and y_new <= sin_y) or (boundary == "upper" and y_new >= sin_y):
                nx, ny = calculate_normal(x_new, A, omega, phi, D)
                U, V = update_velocity(U, V, nx, ny)
                y_new = sin_y
                break

        x, y = x_new, y_new
        yield x, y


class AnimationHandler:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ball = plt.Circle((INIT_PARAMS["x"], INIT_PARAMS["y"]), BALL_RADIUS, color='green')
        self.init_graphics()

    def init_graphics(self):
        """Инициализация графики."""
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor("white")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-150, 150)

        # Границы и синусоиды
        self.ax.axvline(X_MIN, color="red")
        self.ax.axvline(X_MAX, color="red")

        for params in SIN_PARAMS.values():
            x_values = range(-150, 150)
            y_values = [calculate_sin(x, *params.values()) for x in x_values]
            self.ax.plot(x_values, y_values, color="blue")

        self.ax.add_patch(self.ball)

    def update_ball(self, pos):
        """Обновление позиции мяча."""
        x, y = pos
        self.ball.set_center((x, y))
        return self.ball

    def run(self):
        """Запуск анимации."""
        ani = animation.FuncAnimation(
            self.fig,
            self.update_ball,
            get_pos(INIT_PARAMS["x"], INIT_PARAMS["y"], INIT_PARAMS["U"], INIT_PARAMS["V"]),
            interval=30,
            repeat=False
        )
        plt.show()


if __name__ == '__main__':
    AnimationHandler().run()