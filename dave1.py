import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

@dataclass
class Particle:
    """Particule quantique avec spin et position"""
    position: np.ndarray
    momentum: np.ndarray
    spin: float
    mass: float
    charge: float
    trajectory: List[np.ndarray]

class QuantumFieldSimulator:
    """Simulateur de champ quantique avec interactions non-linéaires"""
    
    def __init__(self, grid_size: int = 50, dt: float = 0.01, hbar: float = 1.0):
        self.grid_size = grid_size
        self.dt = dt
        self.hbar = hbar
        self.field = np.random.randn(grid_size, grid_size) + 1j * np.random.randn(grid_size, grid_size)
        self.potential = self._generate_fractal_potential()
        self.particles: List[Particle] = []
        self.energy_history = []
        self.entropy_history = []
        self.norm_history = []
        self.time_history = []
        self.current_time = 0
        
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
        g = 0.5
        return g * np.abs(self.field)**2 * self.field
    
    def add_particle(self, x: float, y: float, vx: float, vy: float):
        """Ajoute une particule avec conditions initiales"""
        position = np.array([x, y])
        momentum = np.array([vx, vy])
        spin = random.choice([-0.5, 0.5])
        particle = Particle(position, momentum, spin, mass=1.0, charge=1.0, trajectory=[position.copy()])
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
            particle.trajectory.append(particle.position.copy())
            if len(particle.trajectory) > 100:
                particle.trajectory.pop(0)
    
    def _compute_entanglement_entropy(self) -> float:
        """Calcule l'entropie d'intrication via décomposition de Schmidt"""
        mid = self.grid_size // 2
        subsystem_a = self.field[:mid, :]
        u, s, vh = np.linalg.svd(subsystem_a, full_matrices=False)
        s_normalized = s / np.linalg.norm(s)
        s_squared = s_normalized ** 2
        entropy = -np.sum(s_squared * np.log(s_squared + 1e-12))
        return entropy
    
    def _compute_energy(self) -> float:
        """Calcule l'énergie totale du système"""
        laplacian = (np.roll(self.field, 1, axis=0) + np.roll(self.field, -1, axis=0) +
                     np.roll(self.field, 1, axis=1) + np.roll(self.field, -1, axis=1) - 
                     4 * self.field) / (self.grid_size / 10)**2
        kinetic_energy = -self.hbar**2 / 2 * np.sum(np.conj(self.field) * laplacian).real
        potential_energy = np.sum(self.potential * np.abs(self.field)**2).real
        return kinetic_energy + potential_energy
    
    def step(self):
        """Un seul pas d'évolution"""
        self.field = self._schrodinger_step()
        self.field += self.dt * self._non_linear_interaction()
        self.field /= np.linalg.norm(self.field)
        self._update_particles()
        
        self.current_time += self.dt
        self.time_history.append(self.current_time)
        self.energy_history.append(self._compute_energy())
        self.entropy_history.append(self._compute_entanglement_entropy())
        self.norm_history.append(np.linalg.norm(self.field))
        
        if len(self.energy_history) > 200:
            self.time_history.pop(0)
            self.energy_history.pop(0)
            self.entropy_history.pop(0)
            self.norm_history.pop(0)

class QuantumVisualizer:
    """Interface graphique temps réel avec statistiques"""
    
    def __init__(self, sim: QuantumFieldSimulator):
        self.sim = sim
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Simulateur Quantique - Temps Réel', fontsize=16, fontweight='bold')
        
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Panneau principal: densité de probabilité
        self.ax_field = self.fig.add_subplot(gs[:2, :2])
        self.ax_field.set_title('Densité de Probabilité |ψ|²')
        self.ax_field.set_xlabel('x')
        self.ax_field.set_ylabel('y')
        
        # Potentiel
        self.ax_potential = self.fig.add_subplot(gs[2, 0])
        self.ax_potential.set_title('Potentiel V(x,y)')
        
        # Phase du champ
        self.ax_phase = self.fig.add_subplot(gs[2, 1])
        self.ax_phase.set_title('Phase arg(ψ)')
        
        # Graphiques temporels
        self.ax_energy = self.fig.add_subplot(gs[0, 2])
        self.ax_energy.set_title('Énergie Totale')
        self.ax_energy.set_xlabel('Temps')
        self.ax_energy.grid(True, alpha=0.3)
        
        self.ax_entropy = self.fig.add_subplot(gs[1, 2])
        self.ax_entropy.set_title('Entropie d\'Intrication')
        self.ax_entropy.set_xlabel('Temps')
        self.ax_entropy.grid(True, alpha=0.3)
        
        # Statistiques textuelles
        self.ax_stats = self.fig.add_subplot(gs[2, 2])
        self.ax_stats.axis('off')
        
        # Initialisation des plots
        self.im_field = None
        self.im_potential = None
        self.im_phase = None
        self.particle_plots = []
        self.trajectory_plots = []
        
    def init_animation(self):
        """Initialise l'animation"""
        probability_density = np.abs(self.sim.field)**2
        self.im_field = self.ax_field.imshow(probability_density, cmap='hot', interpolation='bilinear', origin='lower')
        self.fig.colorbar(self.im_field, ax=self.ax_field, label='|ψ|²')
        
        self.im_potential = self.ax_potential.imshow(self.sim.potential, cmap='viridis', origin='lower')
        self.fig.colorbar(self.im_potential, ax=self.ax_potential, label='V')
        
        phase = np.angle(self.sim.field)
        self.im_phase = self.ax_phase.imshow(phase, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
        self.fig.colorbar(self.im_phase, ax=self.ax_phase, label='arg(ψ)')
        
        return [self.im_field, self.im_potential, self.im_phase]
    
    def update(self, frame):
        """Mise à jour pour chaque frame"""
        # Évolution du système
        self.sim.step()
        
        # Mise à jour densité de probabilité
        probability_density = np.abs(self.sim.field)**2
        self.im_field.set_array(probability_density)
        
        # Mise à jour phase
        phase = np.angle(self.sim.field)
        self.im_phase.set_array(phase)
        
        # Effacer anciennes particules
        for plot in self.particle_plots:
            plot.remove()
        for plot in self.trajectory_plots:
            plot.remove()
        self.particle_plots.clear()
        self.trajectory_plots.clear()
        
        # Dessiner particules et trajectoires
        colors = ['cyan', 'yellow', 'lime', 'magenta', 'orange']
        for i, particle in enumerate(self.sim.particles):
            color = colors[i % len(colors)]
            
            # Trajectoire
            if len(particle.trajectory) > 1:
                traj = np.array(particle.trajectory)
                line, = self.ax_field.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.5, linewidth=1.5)
                self.trajectory_plots.append(line)
            
            # Position actuelle
            marker = 'o' if particle.spin > 0 else 's'
            scatter = self.ax_field.scatter(particle.position[0], particle.position[1], 
                                          c=color, s=100, marker=marker, edgecolors='white', linewidths=2)
            self.particle_plots.append(scatter)
        
        # Mise à jour graphiques temporels
        self.ax_energy.clear()
        self.ax_energy.plot(self.sim.time_history, self.sim.energy_history, 'b-', linewidth=2)
        self.ax_energy.set_title('Énergie Totale')
        self.ax_energy.set_xlabel('Temps')
        self.ax_energy.grid(True, alpha=0.3)
        self.ax_energy.set_xlim(max(0, self.sim.current_time - 2), self.sim.current_time + 0.1)
        
        self.ax_entropy.clear()
        self.ax_entropy.plot(self.sim.time_history, self.sim.entropy_history, 'r-', linewidth=2)
        self.ax_entropy.set_title('Entropie d\'Intrication')
        self.ax_entropy.set_xlabel('Temps')
        self.ax_entropy.grid(True, alpha=0.3)
        self.ax_entropy.set_xlim(max(0, self.sim.current_time - 2), self.sim.current_time + 0.1)
        
        # Statistiques textuelles
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        stats_text = f"""
STATISTIQUES TEMPS RÉEL

Temps: {self.sim.current_time:.3f} s
Particules: {len(self.sim.particles)}
Norme: {self.sim.norm_history[-1]:.6f}

Énergie: {self.sim.energy_history[-1]:.4f}
Entropie: {self.sim.entropy_history[-1]:.4f}

ΔE: {np.std(self.sim.energy_history[-50:]) if len(self.sim.energy_history) > 50 else 0:.5f}
ΔS: {np.std(self.sim.entropy_history[-50:]) if len(self.sim.entropy_history) > 50 else 0:.5f}

Grid: {self.sim.grid_size}×{self.sim.grid_size}
dt: {self.sim.dt}
ℏ: {self.sim.hbar}
        """
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Légende particules
        spin_up = mpatches.Patch(color='white', label='○ = Spin ↑')
        spin_down = mpatches.Patch(color='white', label='□ = Spin ↓')
        self.ax_field.legend(handles=[spin_up, spin_down], loc='upper right', fontsize=8)
        
        return [self.im_field, self.im_phase]
    
    def run(self, frames=500, interval=50):
        """Lance l'animation"""
        anim = FuncAnimation(self.fig, self.update, init_func=self.init_animation,
                           frames=frames, interval=interval, blit=False)
        plt.show()
        return anim

# Exécution
if __name__ == "__main__":
    # Initialisation
    sim = QuantumFieldSimulator(grid_size=64, dt=0.008, hbar=1.0)
    
    # Ajout de particules avec différents spins
    for i in range(6):
        angle = 2 * np.pi * i / 6
        radius = 15
        x = 32 + radius * np.cos(angle)
        y = 32 + radius * np.sin(angle)
        vx = -0.5 * np.sin(angle)
        vy = 0.5 * np.cos(angle)
        sim.add_particle(x, y, vx, vy)
    
    # Lancement de la visualisation
    viz = QuantumVisualizer(sim)
    animation = viz.run(frames=1000, interval=30)
