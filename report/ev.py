import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow

def draw_hex(ax, center, size, color, text=""):
    """Dibuja un hexágono con texto centrado"""
    angles = np.linspace(0, 2*np.pi, 7)
    x = center[0] + size * np.cos(angles)
    y = center[1] + size * np.sin(angles)
    ax.fill(x, y, color, alpha=0.5, edgecolor='black')
    ax.text(center[0], center[1], text, ha='center', va='center', fontweight='bold')

def create_algorithm_evolution():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Título y configuración
    ax.set_title("Evolución de los Algoritmos de Hex", fontsize=16, pad=20)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    # Línea de tiempo
    ax.plot([0, 10], [0, 0], 'k-', linewidth=2)
    
    # Puntos de evolución
    milestones = [
        (1, "A* Básico\n(Heurística simple)", "#FF6B6B", (0.3, 0.2)),
        (3, "Minimax\n(BFS)", "#4ECDC4", (0.3, -0.3)),
        (5, "Minimax Avanzado\n(Heurística compuesta)", "#45B7D1", (0.3, 0.2)),
        (7, "AlephNull\n(Optimizaciones finales)", "#A5D8A2", (0.3, -0.3)),
        (9, "SleepingDragon\n(Híbrido MC, rechazado)", "#FFC154", (0.3, 0.2))
    ]
    
    for x, label, color, (dx, dy) in milestones:
        # Punto en línea de tiempo
        ax.plot(x, 0, 'ko', markersize=10)
        
        # Hexágono con etiqueta
        draw_hex(ax, (x + dx, dy), 0.4, color, label.split('\n')[0])
        
        # Texto descriptivo
        ax.text(x + dx, dy - 0.25, label.split('\n')[1], 
                ha='center', va='top', fontsize=9)
    
    # Flechas de progreso
    for i in range(len(milestones)-1):
        ax.annotate("", xy=(milestones[i+1][0]-0.1, 0), 
                    xytext=(milestones[i][0]+0.1, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Leyenda de colores
    metrics = [
        ("Velocidad (s/mov)", [100, 1, 0.5, 0.1, 5]),
        ("Win Rate (%)", [60, 85, 100, 100, 92]),
        ("Profundidad", [-1, 3, 2, 2, 2])
    ]
    
    for i, (label, values) in enumerate(metrics):
        y_pos = -1.5 - i*0.3
        ax.text(-0.5, y_pos, label, ha='right', va='center')
        
        for j, val in enumerate(values):
            x_pos = milestones[j][0]
            ax.text(x_pos, y_pos, str(val), ha='center', va='center')
            draw_hex(ax, (x_pos, y_pos), 0.15, milestones[j][2])
    
    plt.tight_layout()
    plt.savefig("evolucion_algoritmos.pdf", bbox_inches='tight')
    plt.close()

create_algorithm_evolution()
