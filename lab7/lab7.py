import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math

# --- Класс для треугольника ---
class Triangle:
    def __init__(self, x, y, vx, vy, size, mass, color, ax, speed_multiplier=1.0, angular_multiplier=1.0):
        self.x = x
        self.y = y
        self.vx = vx * speed_multiplier
        self.vy = vy * speed_multiplier
        self.size = size
        self.mass = mass
        self.color = color
        self.angle = 0  # Угол поворота (в градусах)
        self.angular_velocity = 0  # Угловая скорость
        self.angular_multiplier = angular_multiplier
        self.path_x = [x + size / 2]
        self.path_y = [y + self.get_height() / 2]

        # Создаём патч треугольника для Matplotlib
        self.vertices = self.calculate_vertices()
        self.patch = patches.Polygon(
            self.vertices,
            closed=True,
            edgecolor=self.color,
            facecolor=self.color
        )
        ax.add_patch(self.patch)

        # Создаём линию для траектории
        self.line, = ax.plot(self.path_x, self.path_y, color=self.color, linewidth=1, linestyle='--')

    def get_height(self):
        return self.size * (math.sqrt(3) / 2)

    def calculate_vertices(self):
        # Вычисляем вершины треугольника на основе текущих координат и угла
        half_size = self.size / 2
        height = self.get_height()
        points = [
            (self.x + half_size, self.y),
            (self.x, self.y + height),
            (self.x + self.size, self.y + height)
        ]
        # Поворот вокруг центра треугольника
        rotated_points = [self.rotate_point(px, py) for (px, py) in points]
        return rotated_points

    def rotate_point(self, px, py):
        # Поворот точки (px, py) вокруг центра треугольника
        cx = self.x + self.size / 2
        cy = self.y + self.get_height() / 2
        radians = math.radians(self.angle)
        cos_theta = math.cos(radians)
        sin_theta = math.sin(radians)
        dx = px - cx
        dy = py - cy
        qx = cx + cos_theta * dx - sin_theta * dy
        qy = cy + sin_theta * dx + cos_theta * dy
        return (qx, qy)

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.angle = (self.angle + self.angular_velocity * self.angular_multiplier) % 360  # Обновляем угол поворота

        # Ограничиваем угловую скорость
        max_angular_velocity = 5.0
        if self.angular_velocity > max_angular_velocity:
            self.angular_velocity = max_angular_velocity
        elif self.angular_velocity < -max_angular_velocity:
            self.angular_velocity = -max_angular_velocity

        # Добавляем текущую позицию в траекторию
        self.path_x.append(self.x + self.size / 2)
        self.path_y.append(self.y + self.get_height() / 2)

    def check_collision_with_walls(self, width, height):
        height_triangle = self.get_height()
        # Проверка столкновения с левым и правым краем
        if self.x <= 0:
            self.vx = abs(self.vx)
            self.angular_velocity += 0.1  # Вращение при столкновении с левой стеной
        elif self.x + self.size >= width:
            self.vx = -abs(self.vx)
            self.angular_velocity -= 0.1  # Вращение при столкновении с правой стеной

        # Проверка столкновения с верхним и нижним краем
        if self.y <= 0:
            self.vy = abs(self.vy)
            self.angular_velocity += 0.1  # Вращение при столкновении с верхней стеной
        elif self.y + height_triangle >= height:
            self.vy = -abs(self.vy)
            self.angular_velocity -= 0.1  # Вращение при столкновении с нижней стеной

    def check_collision_with_triangle(self, other):
        # Проверка пересечения ось-выравненных bounding box
        if (
            self.x < other.x + other.size and
            self.x + self.size > other.x and
            self.y < other.y + other.get_height() and
            self.y + self.get_height() > other.y
        ):
            # При столкновении меняем угловые скорости с увеличенным коэффициентом
            collision_angular_factor = 0.5  # Увеличенный коэффициент для более заметного вращения
            self.angular_velocity += collision_angular_factor * (self.vx - other.vx)
            other.angular_velocity += collision_angular_factor * (other.vx - self.vx)

            # Обмен скоростями для линейного движения
            self.vx, other.vx = other.vx, self.vx
            self.vy, other.vy = other.vy, self.vy

    def update_patch(self):
        # Обновляем вершины треугольника
        self.vertices = self.calculate_vertices()
        self.patch.set_xy(self.vertices)

    def update_line(self):
        # Ограничиваем длину траектории для оптимизации
        max_length = 1000
        if len(self.path_x) > max_length:
            self.path_x = self.path_x[-max_length:]
            self.path_y = self.path_y[-max_length:]
        # Обновляем линию траектории
        self.line.set_data(self.path_x, self.path_y)

class Simulation:
    def __init__(self, speed_multiplier=2.0, angular_multiplier=5.0):
        self.width = 6
        self.height = 6
        self.triangles = []
        self.speed_multiplier = speed_multiplier  # Коэффициент увеличения скорости
        self.angular_multiplier = angular_multiplier  # Коэффициент увеличения угловой скорости

        # Инициализация Matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('white')  # Устанавливаем белый фон
        plt.title("Лабораторная №7")

        # Задаём начальные данные для треугольников
        self.initialize_triangles()

        # Создаём анимацию
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_animation,
            frames=600,  # Ограничение количества кадров
            interval=16,  # Примерно 60 FPS
            blit=True,
            repeat=False
        )

    def initialize_triangles(self):
        predefined_triangles = [
            {"x": 1, "y": 1, "vx": 0.02, "vy": 0.03, "size": 0.4, "mass": 1, "color": "red"},
            {"x": 3, "y": 2, "vx": -0.03, "vy": 0.02, "size": 0.5, "mass": 2, "color": "green"},
            {"x": 5, "y": 4, "vx": 0.01, "vy": -0.02, "size": 0.3, "mass": 1.5, "color": "blue"}
        ]

        for triangle_data in predefined_triangles:
            triangle = Triangle(
                x=triangle_data["x"],
                y=triangle_data["y"],
                vx=triangle_data["vx"],
                vy=triangle_data["vy"],
                size=triangle_data["size"],
                mass=triangle_data["mass"],
                color=triangle_data["color"],
                ax=self.ax,
                speed_multiplier=self.speed_multiplier,
                angular_multiplier=self.angular_multiplier
            )
            self.triangles.append(triangle)

    def init_animation(self):
        # Инициализация анимации
        artists = []
        for triangle in self.triangles:
            artists.append(triangle.patch)
            artists.append(triangle.line)
        return artists

    def animate(self, frame):
        # Обновление состояния симуляции
        self.update()

        # Обновление патчей и линий для отображения
        artists = []
        for triangle in self.triangles:
            triangle.update_patch()
            triangle.update_line()
            artists.append(triangle.patch)
            artists.append(triangle.line)

        return artists

    def update(self):
        # Обновляем позицию и проверяем столкновения
        for triangle in self.triangles:
            triangle.move()
            triangle.check_collision_with_walls(self.width, self.height)

        # Проверяем столкновения между треугольниками
        for i, triangle1 in enumerate(self.triangles):
            for triangle2 in self.triangles[i + 1:]:
                triangle1.check_collision_with_triangle(triangle2)

    def run(self):
        plt.show()

# --- Точка входа ---
if __name__ == "__main__":
    # Параметры можно настроить по желанию
    simulation = Simulation(speed_multiplier=3.0, angular_multiplier=50.0)
    simulation.run()
