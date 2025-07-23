# nfield_engine.py
# Código base del motor N (módulo núcleo)
# nfield_engine.py

import numpy as np

class NFieldEngine:
    def __init__(self, dim=50):
        self.dim = dim
        self.field = np.random.uniform(0.4, 0.6, (dim, dim))

    def reset(self):
        """Reinicia el campo con valores entre 0.4 y 0.6."""
        self.field = np.random.uniform(0.4, 0.6, (self.dim, self.dim))

    def inject_pattern(self, pattern, x=0, y=0):
        """Inyecta una matriz 'pattern' en el campo en la posición (x, y)."""
        pr, pc = pattern.shape
        for r in range(pr):
            for c in range(pc):
                rr, cc = r + y, c + x
                if 0 <= rr < self.dim and 0 <= cc < self.dim:
                    self.field[rr][cc] = pattern[r, c]

    def diffuse(self, alpha=0.05):
        """Aplica un paso de difusión al campo (modelo de relajación simple)."""
        new_field = self.field.copy()
        for r in range(self.dim):
            for c in range(self.dim):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.dim and 0 <= nc < self.dim:
                            neighbors.append(self.field[nr][nc])
                avg = np.mean(neighbors)
                new_field[r][c] += alpha * (avg - self.field[r][c])
        self.field = new_field

    def export(self):
        """Devuelve el campo como lista de listas (para frontend o API)."""
        return self.field.tolist()

    def evaluate_variance(self):
        """Calcula la varianza global del campo (como métrica básica)."""
        return float(np.var(self.field))
