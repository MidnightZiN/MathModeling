import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
import random

# --- Класс для круга ---
class Circle:
    def __init__(self, x, y, vx, vy, radius, mass, color, ax, speed_multiplier=1.0):
        self.x = x
        self.y = y
        self.vx = vx * speed_multiplier
        self.vy = vy * speed_multiplier
        self.radius = radius
        self.mass = mass
        self.color = color
        self.path_x = [x]
        self.path_y = [y]

        # Создаём патч круга для Matplotlib
        self.patch = patches.Circle(
            (self.x, self.y),
            self.radius,
            edgecolor=self.color,
            facecolor=self.color,
            linewidth=1
        )
        ax.add_patch(self.patch)

        # Создаём линию для траектории (пунктирная линия)
        self.line, = ax.plot(
            self.path_x,
            self.path_y,
            color=self.color,
            linewidth=0.5,
            linestyle='--'
        )

    def move(self):
        self.x += self.vx
        self.y += self.vy

        # Добавляем текущую позицию в траекторию
        self.path_x.append(self.x)
        self.path_y.append(self.y)

    def check_collision_with_walls(self, width, height):
        # Проверка столкновения с левым и правым краем
        if self.x - self.radius <= 0:
            self.x = self.radius  # Корректировка позиции
            self.vx = abs(self.vx)
        elif self.x + self.radius >= width:
            self.x = width - self.radius
            self.vx = -abs(self.vx)

        # Проверка столкновения с верхним и нижним краем
        if self.y - self.radius <= 0:
            self.y = self.radius  # Корректировка позиции
            self.vy = abs(self.vy)
        elif self.y + self.radius >= height:
            self.y = height - self.radius
            self.vy = -abs(self.vy)

    def check_collision_with_circle(self, other):
        # Вычисляем расстояние между центрами двух кругов
        distance = math.hypot(self.x - other.x, self.y - other.y)
        if distance < self.radius + other.radius:
            # Простое отражение (обмен скоростями)
            self.vx, other.vx = other.vx, self.vx
            self.vy, other.vy = other.vy, self.vy

    def update_patch(self):
        # Обновляем позицию круга
        self.patch.center = (self.x, self.y)

    def update_line(self):
        # Ограничиваем длину траектории для оптимизации
        max_length = 1000
        if len(self.path_x) > max_length:
            self.path_x = self.path_x[-max_length:]
            self.path_y = self.path_y[-max_length:]

        # Обновляем линию траектории
        self.line.set_data(self.path_x, self.path_y)

# --- Класс для симуляции ---
class Simulation:
    def __init__(self, speed_multiplier=2.0, num_circles=3):
        self.width = 10
        self.height = 10
        self.circles = []
        self.speed_multiplier = speed_multiplier  # Коэффициент увеличения скорости
        self.num_circles = num_circles  # Количество шариков

        # Инициализация Matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('white')  # Устанавливаем белый фон
        plt.title("Лабораторная №7")

        # Задаём начальные данные для кругов
        self.initialize_circles()

        # Создаём анимацию
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_animation,
            frames=600,      # Ограничение количества кадров
            interval=16,     # Примерно 60 FPS
            blit=True,
            repeat=False
        )

    def initialize_circles(self):
        for _ in range(self.num_circles):
            radius = random.uniform(0.1, 0.3)  # Случайный радиус от 0.1 до 0.3
            x = random.uniform(radius, self.width - radius)  # Случайная позиция по X
            y = random.uniform(radius, self.height - radius)  # Случайная позиция по Y
            vx = random.uniform(-0.03, 0.03)  # Случайная скорость по X
            vy = random.uniform(-0.03, 0.03)  # Случайная скорость по Y
            mass = radius ** 2  # Пример расчёта массы (пропорционально площади)
            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'orange'])  # Случайный цвет

            circle = Circle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                radius=radius,
                mass=mass,
                color=color,
                ax=self.ax,
                speed_multiplier=self.speed_multiplier
            )
            self.circles.append(circle)

    def init_animation(self):
        # Инициализация анимации
        artists = []
        for circle in self.circles:
            artists.append(circle.patch)
            artists.append(circle.line)
        return artists

    def animate(self, frame):
        # Обновление состояния симуляции
        self.update()

        # Обновление патчей и линий для отображения
        artists = []
        for circle in self.circles:
            circle.update_patch()
            circle.update_line()
            artists.append(circle.patch)
            artists.append(circle.line)
        return artists

    def update(self):
        # Обновляем позицию и проверяем столкновения с стенами
        for circle in self.circles:
            circle.move()
            circle.check_collision_with_walls(self.width, self.height)

        # Проверяем столкновения между кругами
        for i, circle1 in enumerate(self.circles):
            for circle2 in self.circles[i + 1:]:
                circle1.check_collision_with_circle(circle2)

    def run(self):
        plt.show()

# --- Точка входа ---
if __name__ == "__main__":
    # Параметры можно настроить по желанию
    # speed_multiplier: увеличивает скорость движения
    # num_circles: количество шариков
    simulation = Simulation(speed_multiplier=9.0, num_circles=5)
    simulation.run()
