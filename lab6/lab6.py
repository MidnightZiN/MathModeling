import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt
from matplotlib.patches import Circle

# Параметры окна
WIN_WIDTH, WIN_HEIGHT = 800, 640
SUN_COLOR = (1.0, 0.0, 0.0)        # Солнце (желтый)
PLANET_COLOR = (0.0, 0.4, 1.0)     # Планета (синий)
TRAIL_COLOR = (0.392, 0.392, 0.392)# След (серый)

# Параметры Солнца
SUN_RADIUS = 15
SUN_MASS = 5000
X0, Y0 = WIN_WIDTH // 2, WIN_HEIGHT // 2

# Параметры планеты
PLANET_RADIUS = 6
CRASH_DIST = 10
OUT_DIST = 1000

# Начальные параметры планеты
x, y = 100.0, 290.0  # Начальная позиция
vx, vy = 0.1, 1.5    # Начальная скорость
ax, ay = 0.0, 0.0    # Начальное ускорение

trail_x = []
trail_y = []

def calculate_acceleration(x, y, mass, center_x, center_y):
    r = sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    ax = mass * (center_x - x) / r ** 3
    ay = mass * (center_y - y) / r ** 3
    return ax, ay

fig, ax_plot = plt.subplots(figsize=(8, 6))
ax_plot.set_xlim(0, WIN_WIDTH)
ax_plot.set_ylim(0, WIN_HEIGHT)
ax_plot.set_aspect('equal', adjustable='box')
ax_plot.set_title("Лаб6")

# Добавляем Солнце
sun = Circle((X0, Y0), SUN_RADIUS, color=SUN_COLOR)
ax_plot.add_patch(sun)

# Линия для следа планеты
trail_line, = ax_plot.plot([], [], color=TRAIL_COLOR, linewidth=1, marker='o', markersize=1, linestyle='None')
planet = Circle((x, y), PLANET_RADIUS, color=PLANET_COLOR)
ax_plot.add_patch(planet)

def init():
    trail_line.set_data([], [])
    planet.center = (x, y)
    return trail_line, planet

def update(frame):
    global x, y, vx, vy, ax, ay, trail_x, trail_y

    # Вычисляем ускорение
    ax, ay = calculate_acceleration(x, y, SUN_MASS, X0, Y0)

    # Обновляем скорость и положение
    vx += ax
    vy += ay
    x += vx
    y += vy

    # Добавляем точку в след
    trail_x.append(x)
    trail_y.append(y)
    if len(trail_x) > 500:
        trail_x.pop(0)
        trail_y.pop(0)

    trail_line.set_data(trail_x, trail_y)
    planet.center = (x, y)

    # Проверка выхода за границы
    if not (0 <= x <= WIN_WIDTH and 0 <= y <= WIN_HEIGHT):
        print("Планета покинула границы экрана.")
        anim.event_source.stop()

    return trail_line, planet

anim = animation.FuncAnimation(fig, update, frames=10000, init_func=init,
                               interval=1, blit=True, repeat=False)

plt.show()
