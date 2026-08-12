import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ==========================================
# LATEX STYLE CONFIGURATION FOR MATPLOTLIB
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",  # Use Computer Modern fonts for math
    "axes.formatter.use_mathtext": True
})


from Gradient_Tracking import GradientTracking

# ==========================================
# GRAPH TOPOLOGY GENERATORS
# ==========================================
def get_path_graph(N):
    adj = np.zeros((N, N))
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    return adj

def get_cycle_graph(N):
    adj = np.zeros((N, N))
    for i in range(N):
        adj[i, (i + 1) % N] = adj[i, (i - 1) % N] = 1
    return adj

def get_star_graph(N):
    adj = np.zeros((N, N))
    for i in range(1, N):
        adj[0, i] = adj[i, 0] = 1
    return adj

def get_complete_graph(N):
    adj = np.ones((N, N)) - np.eye(N)
    return adj

def get_wheel_graph(N):
    adj = get_cycle_graph(N - 1)
    adj = np.pad(adj, ((0, 1), (0, 1)), mode='constant')
    for i in range(N - 1):
        adj[N - 1, i] = adj[i, N - 1] = 1
    return adj


# ==========================================
# 10 UNIQUE 3D TEST CONFIGURATIONS (d = 3)
# ==========================================
TEST_CONFIGS = [
    {
        "name": "Test 1: 3-Agent Path Graph (3D)",
        "N": 3, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_path_graph(3),
        "z_init": np.array([[10.0, -5.0, 2.0], [-8.0, 4.0, 12.0], [3.0, 15.0, -6.0]]),
        "b_points": np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [-5.0, 2.0, 8.0]]),
        "Q_matrices": np.array([
            [[2.0, 0.1, 0.0], [0.1, 1.5, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.2], [0.0, 2.5, 0.0], [0.2, 0.0, 1.8]],
            [[3.0, 0.2, 0.1], [0.2, 1.0, 0.0], [0.1, 0.0, 2.0]]
        ])
    },
    {
        "name": "Test 2: 3-Agent Cycle Graph (3D)",
        "N": 3, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_cycle_graph(3),
        "z_init": np.array([[-12.0, 2.0, 4.0], [6.0, -10.0, -3.0], [0.0, 8.0, 15.0]]),
        "b_points": np.array([[-2.0, -4.0, 6.0], [8.0, 1.0, -2.0], [1.0, 7.0, 3.0]]),
        "Q_matrices": np.array([
            [[1.2, 0.3, 0.0], [0.3, 1.8, 0.1], [0.0, 0.1, 1.0]],
            [[2.1, 0.0, 0.0], [0.0, 1.0, 0.4], [0.0, 0.4, 2.2]],
            [[1.0, 0.1, 0.2], [0.1, 3.0, 0.0], [0.2, 0.0, 1.5]]
        ])
    },
    {
        "name": "Test 3: 3-Agent Star Graph (3D)",
        "N": 3, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_star_graph(3),
        "z_init": np.array([[1.0, 1.0, 1.0], [20.0, -15.0, 5.0], [-10.0, -10.0, -10.0]]),
        "b_points": np.array([[3.0, -3.0, 2.0], [-1.0, 4.0, 9.0], [6.0, 0.0, -4.0]]),
        "Q_matrices": np.array([
            [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
            [[2.5, 0.2, 0.0], [0.2, 1.2, 0.1], [0.0, 0.1, 1.8]],
            [[1.0, 0.3, 0.3], [0.3, 2.0, 0.0], [0.3, 0.0, 2.5]]
        ])
    },
    {
        "name": "Test 4: 4-Agent Path Graph (3D)",
        "N": 4, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_path_graph(4),
        "z_init": np.array([[15.0, 0.0, -5.0], [-5.0, 12.0, 8.0], [2.0, -15.0, 10.0], [0.0, 0.0, -20.0]]),
        "b_points": np.array([[1.0, 2.0, 3.0], [-4.0, 0.0, 5.0], [6.0, -2.0, -1.0], [0.0, 8.0, 4.0]]),
        "Q_matrices": np.array([
            [[2.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.0, 0.2, 0.1], [0.2, 1.5, 0.0], [0.1, 0.0, 1.2]],
            [[3.1, 0.0, 0.4], [0.0, 1.0, 0.0], [0.4, 0.0, 2.0]],
            [[1.8, 0.3, 0.0], [0.3, 2.2, 0.2], [0.0, 0.2, 1.1]]
        ])
    },
    {
        "name": "Test 5: 4-Agent Cycle Graph (3D)",
        "N": 4, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_cycle_graph(4),
        "z_init": np.array([[-10.0, -10.0, 0.0], [10.0, -10.0, 5.0], [10.0, 10.0, -5.0], [-10.0, 10.0, 10.0]]),
        "b_points": np.array([[-2.0, 3.0, 1.0], [4.0, -5.0, 2.0], [0.0, 6.0, -3.0], [-3.0, -1.0, 7.0]]),
        "Q_matrices": np.array([
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.2, 0.1, 0.0], [0.1, 1.1, 0.2], [0.0, 0.2, 1.9]],
            [[1.5, 0.4, 0.0], [0.4, 2.5, 0.0], [0.0, 0.0, 1.0]],
            [[3.0, 0.0, 0.2], [0.0, 1.2, 0.1], [0.2, 0.1, 2.1]]
        ])
    },
    {
        "name": "Test 6: 4-Agent Star Graph (3D)",
        "N": 4, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_star_graph(4),
        "z_init": np.array([[0.0, 0.0, 0.0], [25.0, 5.0, -15.0], [-15.0, 20.0, 10.0], [5.0, -25.0, 18.0]]),
        "b_points": np.array([[5.0, 5.0, -5.0], [-5.0, 2.0, 3.0], [8.0, -1.0, 0.0], [-2.0, -6.0, 8.0]]),
        "Q_matrices": np.array([
            [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5]],
            [[1.2, 0.1, 0.1], [0.1, 1.8, 0.0], [0.1, 0.0, 2.0]],
            [[3.0, 0.2, 0.0], [0.2, 1.0, 0.3], [0.0, 0.3, 1.4]],
            [[1.1, 0.0, 0.2], [0.0, 2.8, 0.0], [0.2, 0.0, 1.6]]
        ])
    },
    {
        "name": "Test 7: 4-Agent Complete Graph (3D)",
        "N": 4, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_complete_graph(4),
        "z_init": np.array([[18.0, -12.0, 6.0], [-6.0, 14.0, -18.0], [22.0, 3.0, 11.0], [-11.0, -9.0, -7.0]]),
        "b_points": np.array([[0.0, 10.0, -2.0], [3.0, -3.0, 4.0], [-7.0, 1.0, 5.0], [2.0, 6.0, -8.0]]),
        "Q_matrices": np.array([
            [[1.7, 0.2, 0.0], [0.2, 2.1, 0.1], [0.0, 0.1, 1.3]],
            [[2.4, 0.0, 0.3], [0.0, 1.6, 0.0], [0.3, 0.0, 2.0]],
            [[1.0, 0.1, 0.0], [0.1, 3.1, 0.2], [0.0, 0.2, 1.5]],
            [[2.0, 0.3, 0.1], [0.3, 1.0, 0.0], [0.1, 0.0, 2.7]]
        ])
    },
    {
        "name": "Test 8: 5-Agent Wheel Graph (3D)",
        "N": 5, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_wheel_graph(5),
        "z_init": np.array([[5.0, 5.0, 5.0], [-15.0, 10.0, 20.0], [12.0, -18.0, -8.0], [-8.0, -12.0, 15.0], [20.0, 15.0, -10.0]]),
        "b_points": np.array([[1.0, -1.0, 2.0], [4.0, 6.0, -3.0], [-5.0, 0.0, 8.0], [2.0, -7.0, 1.0], [-3.0, 4.0, -4.0]]),
        "Q_matrices": np.array([
            [[2.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.4, 0.2, 0.0], [0.2, 1.9, 0.1], [0.0, 0.1, 1.1]],
            [[2.8, 0.0, 0.2], [0.0, 1.2, 0.0], [0.2, 0.0, 2.3]],
            [[1.0, 0.1, 0.3], [0.1, 2.6, 0.0], [0.3, 0.0, 1.7]],
            [[2.2, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 2.1]]
        ])
    },
    {
        "name": "Test 9: 5-Agent Path Graph (3D)",
        "N": 5, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_path_graph(5),
        "z_init": np.array([[-30.0, 10.0, 5.0], [20.0, -20.0, 15.0], [0.0, 25.0, -10.0], [-15.0, -15.0, 20.0], [10.0, 5.0, -25.0]]),
        "b_points": np.array([[-4.0, 2.0, 0.0], [6.0, -1.0, 3.0], [0.0, 5.0, -6.0], [-2.0, -8.0, 4.0], [5.0, 3.0, -2.0]]),
        "Q_matrices": np.array([
            [[1.8, 0.1, 0.1], [0.1, 1.3, 0.0], [0.1, 0.0, 2.2]],
            [[2.5, 0.0, 0.2], [0.0, 2.0, 0.1], [0.2, 0.1, 1.0]],
            [[1.1, 0.3, 0.0], [0.3, 1.7, 0.0], [0.0, 0.0, 2.9]],
            [[3.0, 0.2, 0.1], [0.2, 1.1, 0.2], [0.1, 0.2, 1.4]],
            [[1.5, 0.0, 0.0], [0.0, 2.4, 0.3], [0.0, 0.3, 1.8]]
        ])
    },
    {
        "name": "Test 10: 5-Agent Star Graph (3D)",
        "N": 5, "alpha": 0.05, "max_iterations": 300, "tol_grad": 1e-4,
        "adj_matrix": get_star_graph(5),
        "z_init": np.array([[2.0, -2.0, 2.0], [30.0, 0.0, -20.0], [-20.0, 30.0, 10.0], [10.0, -30.0, 25.0], [-15.0, -15.0, -30.0]]),
        "b_points": np.array([[2.0, 2.0, 2.0], [-6.0, 4.0, -1.0], [7.0, -3.0, 5.0], [-1.0, -5.0, -4.0], [3.0, 8.0, 0.0]]),
        "Q_matrices": np.array([
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            [[1.3, 0.1, 0.0], [0.1, 2.2, 0.2], [0.0, 0.2, 1.5]],
            [[2.1, 0.3, 0.1], [0.3, 1.0, 0.0], [0.1, 0.0, 2.4]],
            [[1.0, 0.0, 0.2], [0.0, 2.7, 0.1], [0.2, 0.1, 1.9]],
            [[2.6, 0.2, 0.0], [0.2, 1.4, 0.3], [0.0, 0.3, 1.1]]
        ])
    }
]

# ==========================================
# HELPER FUNCTIONS FOR TEXT REPORT (.cfg)
# ==========================================
def matrix_to_string(mat):
    """Formats a NumPy array to a clean string with 2 decimal places."""
    return np.array2string(mat, formatter={'float_kind': lambda x: f"{x:6.2f}"})

def generate_simulation_report(config, sim_number, converged, actual_iters, max_iters, optimal_cost, final_cost, z_star, final_z):
    """Generates a text block for the .cfg file containing both inputs and outputs."""
    text = f"SIMULATION {sim_number}: {config['name']}\n"
    text += f"{'='*70}\n"
    text += f"Graph Pattern   : {config['name'].split(':')[1].strip()}\n"
    text += f"Agents (N)      : {config['N']}\n"
    text += f"Step-Size (a)   : {config['alpha']}\n"
    text += f"Max Iterations  : {config['max_iterations']}\n"
    text += f"Grad Tolerance  : {config['tol_grad']}\n\n"
    
    text += "--- INPUT: AGENT CONFIGURATIONS ---\n"
    for i in range(config["N"]):
        text += f"[Agent {i+1}]\n"
        text += f"  Initial Pos (z0): {np.round(config['z_init'][i], 2).tolist()}\n"
        text += f"  Target (b_i)    : {np.round(config['b_points'][i], 2).tolist()}\n"
        
        q_lines = matrix_to_string(config['Q_matrices'][i]).split('\n')
        text += f"  Matrix (Q_i)    : {q_lines[0]}\n"
        for line in q_lines[1:]:
            text += f"                    {line}\n"
        text += "\n"

    text += "--- OUTPUT: RESULTS ---\n"
    status = "YES" if converged else "NO (Reached Max Iterations)"
    text += f"Converged successfully? : {status} (Iterations: {actual_iters}/{max_iters})\n\n"
    
    text += f"Optimal Minimum (f*)    : {optimal_cost:.6e}\n"
    text += f"Final Reached Cost (f)  : {final_cost:.6e}\n"
    text += f"Cost Error |f - f*|     : {abs(final_cost - optimal_cost):.6e}\n\n"
    
    text += f"Theoretical Argmin (z*) : {np.round(z_star, 4).tolist()}\n"
    text += f"Final Reached Argmin    : {np.round(final_z, 4).tolist()}\n"
    text += f"Position Error ||z-z*|| : {np.linalg.norm(final_z - z_star):.6e}\n"
    text += f"{'='*70}\n\n"
    
    return text


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def run_simulations():
    # Setup directories
    base_dir = os.path.dirname(__file__)
    report_dir = os.path.abspath(os.path.join(base_dir, '..', 'report'))
    figs_dir = os.path.join(report_dir, 'figs')
    
    os.makedirs(figs_dir, exist_ok=True)
    cfg_path = os.path.join(report_dir, 'Gradient_Tracking.cfg')
    
    print(f"Starting {len(TEST_CONFIGS)} Gradient Tracking Simulations...\n")
    
    # Open the central .cfg file to write the text report
    with open(cfg_path, 'w', encoding='utf-8') as cfg_file:
        cfg_file.write("======================================================================\n")
        cfg_file.write(" GRADIENT TRACKING SIMULATIONS - COMPLETE INPUT/OUTPUT REPORT\n")
        cfg_file.write("======================================================================\n\n")

        for idx, config in enumerate(TEST_CONFIGS):
            sim_number = idx + 1
            print(f"[{sim_number}/{len(TEST_CONFIGS)}] Running: {config['name']}...")
            
            # 1. Setup Algorithm
            gt = GradientTracking(
                z_init=config["z_init"],
                b_points=config["b_points"],
                Q_matrices=config["Q_matrices"],
                adj_matrix=config["adj_matrix"],
                alpha=config["alpha"]
            )

            # 2. Calculate Theoretical Minimum
            Q_sum = np.sum(config["Q_matrices"], axis=0)
            Qb_sum = np.sum([config["Q_matrices"][i] @ config["b_points"][i] for i in range(config["N"])], axis=0)
            z_star = np.linalg.solve(Q_sum, Qb_sum)

            optimal_cost = 0.5 * sum((z_star - config["b_points"][i]) @ config["Q_matrices"][i] @ (z_star - config["b_points"][i]) for i in range(config["N"]))

            # 3. Iteration Loop
            costs_history = [gt.get_global_cost()]
            grad_norms_history = [gt.get_global_gradient_norm()]
            
            max_iter = config.get("max_iterations", 300)
            tol_grad = config.get("tol_grad", 1e-4)
            converged = False

            for _ in range(max_iter):
                gt.step()
                current_cost = gt.get_global_cost()
                current_grad = gt.get_global_gradient_norm()
                
                costs_history.append(current_cost)
                grad_norms_history.append(current_grad)

                if current_grad < tol_grad:
                    converged = True
                    break
                    
            actual_iterations = len(costs_history) - 1
            final_cost = costs_history[-1]
            final_z_mean = np.mean(gt.X, axis=0)
            
            # 4. Generate the Plots (Only charts, no text overlays)
            fig, (ax_cost, ax_grad) = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor('#ffffff')
            
            # Cost Plot
            ax_cost.plot(costs_history, color='#0284c7', linewidth=2.5)
            ax_cost.set_yscale('log')
            ax_cost.set_title('Evolution of the Cost Function $f(z)$', fontsize=14)
            ax_cost.set_xlabel('Iteration ($k$)', fontsize=12)
            ax_cost.set_ylabel('Cost (Log Scale)', fontsize=12)
            ax_cost.grid(True, linestyle='--', alpha=0.6)
            
            # Gradient Norm Plot
            ax_grad.plot(grad_norms_history, color='#ea580c', linewidth=2.5)
            ax_grad.set_yscale('log')
            ax_grad.set_title('Evolution of Gradient Norm $||\\nabla f(z)||$', fontsize=14)
            ax_grad.set_xlabel('Iteration ($k$)', fontsize=12)
            ax_grad.set_ylabel('$||\\nabla f(z)||$ (Log Scale)', fontsize=12)
            ax_grad.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            
            # Save the figure
            save_path = os.path.join(figs_dir, f'gradient_tracking_simulation_{sim_number}.png')
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            # 5. Write data to the shared .cfg file
            report_text = generate_simulation_report(
                config, sim_number, converged, actual_iterations, max_iter, 
                optimal_cost, final_cost, z_star, final_z_mean
            )
            cfg_file.write(report_text)
            
            print(f"  -> Saved figure: {save_path}")

    print(f"\n[✓] All simulations completed! Report saved to: {cfg_path}")


if __name__ == "__main__":
    run_simulations()