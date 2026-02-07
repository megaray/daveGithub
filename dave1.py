import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass
from functools import lru_cache
import random

@dataclass
class Particle:
    """Particule quantique avec spin et position"""
    position: np.ndarray
    momentum: np.ndarray
    spin: float
    mass: float
    charge: float

class QuantumFieldSimulator:
    """Simulateur de champ quantique avec interactions non-linéaires"""
    
    def __init__(self, grid_size: int = 50, dt: float = 0.01, hbar: float = 1.0):
        self.grid_size = grid_size
        self.dt = dt
        self.hbar = hbar
        self.field = np.random.randn(grid_size, grid_size) + 1j * np.random.randn(grid_size, grid_size)
        self.potential = self._generate_fractal_potential()
        self.particles: List[Particle] = []
        
    def _generate_fractal_potential(self) -> np.ndarray:
        """Génère un potentiel fractal avec bruit de Perlin"""
        x = np.linspace(0, 4 * np.pi, self.grid_size)
        y = np.linspace(0, 4 * np.pi, self.grid_size)
        X, Y = np.meshgrid(x, y)
        potential = np.zeros_like(X)
        for octave in range(5):
            freq = 2 ** octave
            potential += np.sin(freq * X) * np.cos(freq * Y) / (freq ** 1.5)
        return potential
    
    @lru_cache(maxsize=128)
    def _cached_hamiltonian_eigenvalue(self, n: int, m: int) -> float:
        """Calcule les valeurs propres de l'hamiltonien avec cache"""
        return (n**2 + m**2) * np.pi**2 / (2 * self.grid_size**2)
    
    def _schrodinger_step(self) -> np.ndarray:
        """Évolution temporelle via l'équation de Schrödinger"""
        laplacian = (np.roll(self.field, 1, axis=0) + np.roll(self.field, -1, axis=0) +
                     np.roll(self.field, 1, axis=1) + np.roll(self.field, -1, axis=1) - 
                     4 * self.field) / (self.grid_size / 10)**2
        kinetic = -self.hbar**2 / 2 * laplacian
        potential_energy = self.potential * self.field
        hamiltonian = kinetic + potential_energy
        evolution = np.exp(-1j * self.dt * hamiltonian / self.hbar)
        return self.field * evolution
    
    def _non_linear_interaction(self) -> np.ndarray:
        """Terme d'interaction non-linéaire de type Gross-Pitaevskii"""
        g = 0.5  # constante de couplage
        return g * np.abs(self.field)**2 * self.field
    
    def add_particle(self, x: float, y: float, vx: float, vy: float):
        """Ajoute une particule avec conditions initiales"""
        position = np.array([x, y])
        momentum = np.array([vx, vy])
        spin = random.choice([-0.5, 0.5])
        particle = Particle(position, momentum, spin, mass=1.0, charge=1.0)
        self.particles.append(particle)
    
    def _update_particles(self):
        """Met à jour les positions des particules via forces du champ"""
        for particle in self.particles:
            ix, iy = int(particle.position[0] % self.grid_size), int(particle.position[1] % self.grid_size)
            force_x = -np.gradient(self.potential, axis=0)[ix, iy]
            force_y = -np.gradient(self.potential, axis=1)[ix, iy]
            acceleration = np.array([force_x, force_y]) / particle.mass
            particle.momentum += acceleration * self.dt
            particle.position += particle.momentum * self.dt / particle.mass
            particle.position %= self.grid_size
    
    def _compute_entanglement_entropy(self) -> float:
        """Calcule l'entropie d'intrication via décomposition de Schmidt"""
        mid = self.grid_size // 2
        subsystem_a = self.field[:mid, :]
        u, s, vh = np.linalg.svd(subsystem_a, full_matrices=False)
        s_normalized = s / np.linalg.norm(s)
        s_squared = s_normalized ** 2
        entropy = -np.sum(s_squared * np.log(s_squared + 1e-12))
        return entropy
    
    def evolve(self, steps: int = 100) -> Tuple[np.ndarray, List[float]]:
        """Fait évoluer le système sur plusieurs pas de temps"""
        entropies = []
        for _ in range(steps):
            self.field = self._schrodinger_step()
            self.field += self.dt * self._non_linear_interaction()
            self.field /= np.linalg.norm(self.field)  # normalisation
            self._update_particles()
            if _ % 10 == 0:
                entropies.append(self._compute_entanglement_entropy())
        return self.field, entropies

# Exécution principale avec analyse
if __name__ == "__main__":
    sim = QuantumFieldSimulator(grid_size=64, dt=0.005)
    
    # Ajout de particules avec distribution gaussienne
    for _ in range(5):
        x, y = np.random.randn(2) * 10 + 32
        vx, vy = np.random.randn(2) * 0.5
        sim.add_particle(x, y, vx, vy)
    
    final_field, entropy_evolution = sim.evolve(steps=200)
    
    print(f"Norme finale du champ: {np.linalg.norm(final_field):.6f}")
    print(f"Entropie d'intrication finale: {entropy_evolution[-1]:.6f}")
    print(f"Nombre de particules simulées: {len(sim.particles)}")
    
#davidou888
print("J'apprend git!!!")

print("j'update mon python ici !")
