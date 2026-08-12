import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from DAS1dot3 import GradientTrackingLogReg
import DAS1dot2

# Matplotlib styling shared by the saved and interactive figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",  # Computer Modern math glyphs
    "axes.formatter.use_mathtext": True
})


# Graph topology generators
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


# Synthetic datasets and heterogeneous agent splits
def generate_synthetic_data(label, M=1000):

    rng = np.random.default_rng(77)
    dataset = rng.uniform(
        low = [-2.0, -2.0],
        high = [2.0, 2.0],
        size = (M,2)
    )
    x1 = dataset[:, 0]
    x2 = dataset[:, 1]
    cubic_mapping = np.column_stack([
        x1,
        x2,
        x1**3
    ])

    superellipse_mapping = np.column_stack([
        x1,
        x2,
        x1**4,
        x2**4
    ])

    cubic_weights = np.array([2.0, -2.0, 3.0])
    superellipse_weights = np.array([1.0, 1.0, -0.05, -0.05])
    bias = 0.5

    cubic_labels = np.where(cubic_mapping @ cubic_weights + bias >= 0, 1, -1)
    ellipse_labels = np.where(
        superellipse_mapping @ superellipse_weights + bias >= 0,
        1,
        -1
    )

    if label == 'Cubic':
        return dataset, cubic_mapping, cubic_labels
    if label == 'SuperEllipse':
        return dataset, superellipse_mapping, ellipse_labels
    raise ValueError(f"Unsupported dataset label: {label!r}")


def distribute_data(N, G=7, M=1000, label='Cubic'):
    """
    Distribute data among agents for gradient tracking.
    Parameters:
    - G: int -> Group number
    - N: int -> Number of agents 
    """
    P = 40 + (G % 3) * 10

    sorting_feature_index = 0

    dataset, mapping, labels = generate_synthetic_data(label=label, M=M)

    baseline_count = M * P//100
    rnd_generator = np.random.default_rng(77)
    all_indices = rnd_generator.permutation(M)
    baseline_indices = all_indices[:baseline_count]
    remaining_indices = all_indices[baseline_count:]

    # The random baseline gives every agent both classes. The sorted remainder
    # creates different feature distributions across the local datasets.
    for _ in range(100):
        rnd_generator.shuffle(baseline_indices)
        tempblocks = np.array_split(baseline_indices, N)

        if all(len(np.unique(labels[block])) == 2 for block in tempblocks):
            datablocks = tempblocks
            break
    else:
        raise RuntimeError(
            "Could not give every agent both label classes after 100 shuffles."
        )

    remaining_feature_values = dataset[
        remaining_indices,
        sorting_feature_index
    ]
    sorting_order = np.argsort(remaining_feature_values)
    sorted_remaining_indices = remaining_indices[sorting_order]

    featurebased_block = np.array_split(sorted_remaining_indices, N)

    local_features = []
    local_labels = []

    for agent in range(N):
        # Each local dataset combines balanced random data with one sorted block.
        combined_indices = np.concatenate((datablocks[agent], featurebased_block[agent]))
        local_features.append(mapping[combined_indices])
        local_labels.append(labels[combined_indices])
    
    return dataset, mapping, labels, local_features, local_labels


def run_task1dot2_comparison(
    mapping,
    labels,
    stepsize,
    max_iterations,
    stopping_threshold
):
    """Run centralized Task 1.2 gradient descent on the comparison dataset."""
    weights = np.zeros(mapping.shape[1])
    bias = 0.0
    costs = []
    gradient_norms = []
    converged = False

    for _ in range(max_iterations + 1):
        # Reusing Task 1.2's objective keeps the centralized comparison aligned.
        cost, grad_w, grad_b = DAS1dot2.LogRegCostFunction(
            mapping,
            labels,
            weights,
            bias
        )
        gradient_norm = float(np.linalg.norm(np.append(grad_w, grad_b)))
        costs.append(float(cost))
        gradient_norms.append(gradient_norm)

        if gradient_norm < stopping_threshold:
            converged = True
            break
        if len(costs) > max_iterations:
            break

        weights -= stepsize * grad_w
        bias -= stepsize * grad_b

    parameters = np.append(weights, bias)
    predictions = np.where(mapping @ weights + bias >= 0, 1, -1)

    return {
        "costs": costs,
        "gradient_norms": gradient_norms,
        "converged": converged,
        "iterations": len(costs) - 1,
        "parameters": parameters,
        "misclassified_percentage": 100.0 * float(np.mean(predictions != labels))
    }

# Logistic-regression simulation configurations
TEST_CONFIGS = [
    {
        "name": "Test 1: 4-Agent Complete Graph. Cubic Mapping",
        "N": 4, "alpha": 0.01, "max_iterations": 1000, "tol_grad": 1e-4,
        "adj_matrix": get_complete_graph(4),
        "z_init": np.array([[10.0, -5.0, 2.0, 1.0], [-8.0, 4.0, 12.0, -1.0], [3.0, 15.0, -6.0, 2.0], [1.0, 2.0, 3.0, -2.0]]),
        "label": "Cubic",
        "M": 1000
    },
    {
        "name": "Test 2: 3-Agent Cycle Graph. Cubic Mapping",
        "N": 3, "alpha": 0.01, "max_iterations": 1000, "tol_grad": 1e-4,
        "adj_matrix": get_cycle_graph(3),
        "z_init": np.array([[-1.0, 2.0, 4.0, 0.0], [6.0, -3.0, -3.0, 0.0], [0.0, 2.0, 1.0, 0.0]]),
        "label": "Cubic",
        "M": 1000
    },
    {
        "name": "Test 3: 4-Agent Star Graph. Super-Ellipse Mapping",
        "N": 4, "alpha": 0.001, "max_iterations": 1000, "tol_grad": 1e-4,
        "adj_matrix": get_star_graph(4),
        "z_init": np.array([[1.0, 1.0, 1.0, 1.0, 0.0], [2.0, -1.5, 5.0, -5.0, 0.0], [-10.0, -1.0, 0.0, 10.0, 0.0], [5.0, -2.0, 1.0, -3.0, 0.0]]),
        "label": "SuperEllipse",
        "M": 1000
    },
    {
        "name": "Test 4: 4-Agent Complete Graph. Super-Ellipse Mapping",
        "N": 4, "alpha": 0.001, "max_iterations": 1000, "tol_grad": 1e-4,
        "adj_matrix": get_complete_graph(4),
        "z_init": np.array([[15.0, 0.0, -5.0, 0.0, 0.0], [-5.0, 12.0, 8.0, 0.0, 0.0], [2.0, -15.0, 10.0, 0.0, 0.0], [0.0, 0.0, -20.0, 0.0, 0.0]]),
        "label": "SuperEllipse",
        "M": 1000
    },
    {
        "name": "Test 5: 3-Agent Cycle Graph. Cubic Mapping",
        "N": 3, "alpha": 0.05, "max_iterations": 1000, "tol_grad": 1e-4,
        "adj_matrix": get_cycle_graph(3),
        "z_init": np.array([[-10.0, -10.0, 0.0, 0.0], [10.0, -10.0, 5.0, 0.0], [10.0, 10.0, -5.0, 0.0]]),
        "label": "Cubic",
        "M": 500
    },
    {
        "name": "Test 6: 4-Agent Path Graph. Super-Ellipse Mapping",
        "N": 4, "alpha": 0.001, "max_iterations": 1000, "tol_grad": 1e-4,
        "adj_matrix": get_path_graph(4),
        "z_init": np.array([[0.0, 0.0, 0.0, 2.0, 0.0], [25.0, 5.0, -15.0, -5.0, 0.0], [-15.0, 20.0, 10.0, 0.0, 0.0], [5.0, -25.0, 18.0, 2.0, 0.0]]),
        "label": "SuperEllipse",
        "M": 500
    },
    
]
# Plain-text result reporting
def generate_simulation_report(
    config,
    sim_number,
    local_features,
    gt_results,
    task1dot2_results
):
    """Build a text comparison of distributed Task 1.3 and Task 1.2."""
    text = f"SIMULATION {sim_number}: {config['name']}\n"
    text += f"{'=' * 78}\n"
    text += f"Dataset              : {config['label']} ({config['M']} samples)\n"
    text += f"Graph Pattern         : {config['name'].split(':', 1)[1].strip()}\n"
    text += f"Agents (N)            : {config['N']}\n"
    text += f"Parameter Dimension   : {config['z_init'].shape[1]} (including bias)\n"
    text += f"Step Size (alpha)     : {config['alpha']}\n"
    text += f"Max Iterations        : {config['max_iterations']}\n"
    text += f"Gradient Tolerance    : {config['tol_grad']}\n\n"

    text += "--- INPUT: AGENT CONFIGURATIONS ---\n"
    for agent in range(config["N"]):
        text += f"[Agent {agent + 1}]\n"
        text += f"  Local samples       : {len(local_features[agent])}\n"
        text += (
            f"  Initial parameters  : "
            f"{np.round(config['z_init'][agent], 4).tolist()}\n"
        )

    gt_status = "YES" if gt_results["converged"] else "NO"
    central_status = "YES" if task1dot2_results["converged"] else "NO"

    text += "\n--- TASK 1.3: DISTRIBUTED GRADIENT TRACKING ---\n"
    text += f"Converged             : {gt_status}\n"
    text += f"Iterations            : {gt_results['iterations']}\n"
    text += f"Final cost            : {gt_results['costs'][-1]:.6e}\n"
    text += f"Final gradient norm   : {gt_results['gradient_norms'][-1]:.6e}\n"
    text += f"Final consensus error : {gt_results['consensus_errors'][-1]:.6e}\n"
    text += f"Misclassified points  : {gt_results['misclassified_percentage']:.4f}%\n"
    text += (
        f"Mean parameters       : "
        f"{np.round(gt_results['parameters'], 6).tolist()}\n"
    )

    text += "\n--- TASK 1.2: CENTRALIZED GRADIENT DESCENT ---\n"
    text += f"Converged             : {central_status}\n"
    text += f"Iterations            : {task1dot2_results['iterations']}\n"
    text += f"Final cost            : {task1dot2_results['costs'][-1]:.6e}\n"
    text += (
        f"Final gradient norm   : "
        f"{task1dot2_results['gradient_norms'][-1]:.6e}\n"
    )
    text += (
        f"Misclassified points  : "
        f"{task1dot2_results['misclassified_percentage']:.4f}%\n"
    )
    text += (
        f"Parameters            : "
        f"{np.round(task1dot2_results['parameters'], 6).tolist()}\n"
    )

    text += "\n--- DIRECT COMPARISON (TASK 1.3 MINUS TASK 1.2) ---\n"
    text += (
        f"Final cost difference : "
        f"{gt_results['costs'][-1] - task1dot2_results['costs'][-1]:.6e}\n"
    )
    text += (
        f"Misclassification diff: "
        f"{gt_results['misclassified_percentage'] - task1dot2_results['misclassified_percentage']:.4f}%\n"
    )
    text += f"{'=' * 78}\n\n"
    return text


def run_simulations():
    base_dir = os.path.dirname(__file__)
    report_dir = os.path.join(base_dir, 'report')
    figs_dir = os.path.join(report_dir, 'figs')

    os.makedirs(figs_dir, exist_ok=True)
    cfg_path = os.path.join(report_dir, 'Gradient_Tracking.cfg')
    report_sections = []

    print(f"Starting {len(TEST_CONFIGS)} Task 1.3 / Task 1.2 comparisons...\n")

    for idx, config in enumerate(TEST_CONFIGS):
        sim_number = idx + 1
        print(f"[{sim_number}/{len(TEST_CONFIGS)}] Running: {config['name']}...")

        dataset, mapping, labels, local_features, local_labels = distribute_data(
            G=7,
            N=config["N"],
            M=config["M"],
            label=config["label"]
        )

        expected_shape = (config["N"], mapping.shape[1] + 1)
        if config["z_init"].shape != expected_shape:
            raise ValueError(
                f"{config['name']} has z_init shape {config['z_init'].shape}; "
                f"expected {expected_shape} for {config['label']}."
            )
        if config["adj_matrix"].shape != (config["N"], config["N"]):
            raise ValueError(
                f"{config['name']} adjacency matrix does not match N={config['N']}."
            )
        if len(dataset) != config["M"]:
            raise ValueError("Generated dataset size does not match the configuration.")

        gt = GradientTrackingLogReg(
            z_init=config["z_init"],
            feature_subsets=local_features,
            label_subsets=local_labels,
            adj_matrix=config["adj_matrix"],
            alpha=config["alpha"],
            tol=config["tol_grad"]
        )

        costs_history = [gt.get_global_cost()]
        grad_norms_history = [gt.get_global_gradient_norm()]
        consensus_errors = [
            float(np.max(np.linalg.norm(gt.X - np.mean(gt.X, axis=0), axis=1)))
        ]

        max_iter = config.get("max_iterations", 300)
        tol_grad = config.get("tol_grad", 1e-4)
        converged = False

        for _ in range(max_iter):
            gt.step()
            current_cost = gt.get_global_cost()
            current_grad = gt.get_global_gradient_norm()
            current_consensus = float(
                np.max(np.linalg.norm(gt.X - np.mean(gt.X, axis=0), axis=1))
            )

            costs_history.append(current_cost)
            grad_norms_history.append(current_grad)
            consensus_errors.append(current_consensus)

            if current_grad < tol_grad and current_consensus < tol_grad:
                converged = True
                break

        actual_iterations = len(costs_history) - 1
        final_parameters = np.mean(gt.X, axis=0)
        features_with_bias = np.column_stack((mapping, np.ones(len(mapping))))
        gt_scores = features_with_bias @ final_parameters
        gt_predictions = np.where(gt_scores >= 0, 1, -1)
        gt_results = {
            "costs": costs_history,
            "gradient_norms": grad_norms_history,
            "consensus_errors": consensus_errors,
            "converged": converged,
            "iterations": actual_iterations,
            "parameters": final_parameters,
            "misclassified_percentage": 100.0 * float(
                np.mean(gt_predictions != labels)
            )
        }

        task1dot2_results = run_task1dot2_comparison(
            mapping=mapping,
            labels=labels,
            stepsize=config["alpha"],
            max_iterations=max_iter,
            stopping_threshold=tol_grad
        )
        task1dot2_scores = features_with_bias @ task1dot2_results["parameters"]

        fig, (ax_cost, ax_grad, ax_boundary) = plt.subplots(
            1,
            3,
            figsize=(16, 5)
        )
        fig.patch.set_facecolor('#ffffff')

        ax_cost.plot(
            costs_history,
            color='#0284c7',
            linewidth=2.2,
            label='Task 1.3 distributed'
        )
        ax_cost.plot(
            task1dot2_results["costs"],
            color='#15803d',
            linewidth=2.0,
            linestyle='--',
            label='Task 1.2 centralized'
        )
        ax_cost.set_title('Cost vs Iterations', fontsize=14)
        ax_cost.set_xlabel('Iteration ($k$)', fontsize=12)
        ax_cost.set_ylabel('Logistic Cost', fontsize=12)
        ax_cost.grid(True, linestyle='--', alpha=0.6)
        ax_cost.legend()

        ax_grad.plot(
            grad_norms_history,
            color='#ea580c',
            linewidth=2.2,
            label='Task 1.3 distributed'
        )
        ax_grad.plot(
            task1dot2_results["gradient_norms"],
            color='#7e22ce',
            linewidth=2.0,
            linestyle='--',
            label='Task 1.2 centralized'
        )
        ax_grad.set_title('Gradient Norm vs Iterations', fontsize=14)
        ax_grad.set_xlabel('Iteration ($k$)', fontsize=12)
        ax_grad.set_ylabel('$||\\nabla f(z)||$', fontsize=12)
        ax_grad.grid(True, linestyle='--', alpha=0.6)
        ax_grad.legend()

        ax_boundary.scatter(
            dataset[:, 0],
            dataset[:, 1],
            c=labels,
            cmap='bwr',
            alpha=0.5,
            s=18,
            edgecolors='none'
        )
        ax_boundary.tricontour(
            dataset[:, 0],
            dataset[:, 1],
            gt_scores,
            levels=[0],
            colors='#0284c7',
            linewidths=2.2
        )
        ax_boundary.tricontour(
            dataset[:, 0],
            dataset[:, 1],
            task1dot2_scores,
            levels=[0],
            colors='#15803d',
            linewidths=2.0,
            linestyles='--'
        )
        ax_boundary.set_title(
            f"{config['label']} Decision Boundaries",
            fontsize=14
        )
        ax_boundary.set_xlabel('$x_1$', fontsize=12)
        ax_boundary.set_ylabel('$x_2$', fontsize=12)
        ax_boundary.set_xlim(-2.0, 2.0)
        ax_boundary.set_ylim(-2.0, 2.0)
        ax_boundary.grid(True, linestyle='--', alpha=0.4)
        ax_boundary.legend(handles=[
            Line2D([0], [0], color='#0284c7', linewidth=2.2, label='Task 1.3'),
            Line2D(
                [0],
                [0],
                color='#15803d',
                linewidth=2.0,
                linestyle='--',
                label='Task 1.2'
            )
        ])

        fig.suptitle(config["name"], fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.94))

        figure_filename = f'gradient_tracking_simulation_{sim_number}.png'
        save_path = os.path.join(figs_dir, figure_filename)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

        report_sections.append(generate_simulation_report(
            config,
            sim_number,
            local_features,
            gt_results,
            task1dot2_results
        ))

        print(f"\nThese are the final Task 1.3 results for {config['label']}:")
        print(
            f"  Final cost: {gt_results['costs'][-1]:.6e}, "
            f"iterations: {gt_results['iterations']}, "
            f"gradient norm: {gt_results['gradient_norms'][-1]:.6e}"
        )
        print(f"  Consensus error: {gt_results['consensus_errors'][-1]:.6e}")
        print(
            f"  Percentage of misclassified points: "
            f"{gt_results['misclassified_percentage']:.2f}%"
        )
        print(f"These are the final Task 1.2 results for {config['label']}:")
        print(
            f"  Final cost: {task1dot2_results['costs'][-1]:.6e}, "
            f"iterations: {task1dot2_results['iterations']}, "
            f"gradient norm: {task1dot2_results['gradient_norms'][-1]:.6e}"
        )
        print(
            f"  Percentage of misclassified points: "
            f"{task1dot2_results['misclassified_percentage']:.2f}%"
        )
        print(f"  Saved comparison figure: {save_path}")

        interactive_framework = getattr(
            fig.canvas,
            "required_interactive_framework",
            None
        )
        if interactive_framework:
            print("  Close the three-panel plot to see the consensus error plot.")
            plt.show(block=True)
        else:
            print("  Non-interactive backend detected; saving plots automatically.")
        plt.close(fig)

#Consensus error plots 
        consensus_fig, consensus_ax = plt.subplots(figsize=(9, 5))
        consensus_ax.plot(
            np.maximum(consensus_errors, np.finfo(float).tiny),
            color='#be123c',
            linewidth=2.2,
            label='Task 1.3 consensus error'
        )
        consensus_ax.axhline(
            tol_grad,
            color='#475569',
            linewidth=1.6,
            linestyle='--',
            label=f'Tolerance ({tol_grad:g})'
        )
        consensus_ax.set_yscale('log')
        consensus_ax.set_title(
            f"{config['name']}\nConsensus Error vs Iterations",
            fontsize=14
        )
        consensus_ax.set_xlabel('Iteration ($k$)', fontsize=12)
        consensus_ax.set_ylabel('Maximum Agent Distance (Log Scale)', fontsize=12)
        consensus_ax.grid(True, linestyle='--', alpha=0.6)
        consensus_ax.legend()
        consensus_fig.tight_layout()

        consensus_filename = f'gradient_tracking_consensus_{sim_number}.png'
        consensus_save_path = os.path.join(figs_dir, consensus_filename)
        consensus_fig.savefig(consensus_save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved consensus figure: {consensus_save_path}")

        if interactive_framework:
            print("  Close the consensus plot to continue to the next simulation.\n")
            plt.show(block=True)
        else:
            print("  Continuing to the next simulation.\n")
        plt.close(consensus_fig)

    cfg_header = (
        "==============================================================================\n"
        " TASK 1.3 GRADIENT TRACKING VS TASK 1.2 CENTRALIZED LOGISTIC REGRESSION\n"
        "==============================================================================\n\n"
    )
    with open(cfg_path, 'w', encoding='utf-8') as cfg_file:
        cfg_file.write(cfg_header + "".join(report_sections))

    print("[OK] All comparisons and plot windows completed.")
    print(f"CFG report saved to: {cfg_path}")

if __name__ == "__main__":
    run_simulations()
