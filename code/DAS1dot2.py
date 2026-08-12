import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

M = 1000
rng = np.random.default_rng(77)
dataset = rng.uniform(
    low = [-2.0, -2.0],
    high = [2.0, 2.0],
    size = (M,2)
)
x1 = dataset[:, 0]
x2 = dataset[:, 1]
CubicMapping = np.column_stack([
    x1,
    x2,
    x1**3
])

SuperEllipseMapping = np.column_stack([
    x1,
    x2,
    x1**4,
    x2**4
])

EllipseLabels = np.zeros(M)

CubicWeights = np.array([2.0, -2.0, 3.0])
SuperEllipseWeights = np.array([1.0, 1.0, -0.05, -0.05])
bias = 0.5

CubicLabels = np.where(CubicMapping @ CubicWeights + bias >= 0, 1, -1)

EllipseLabels = np.where(SuperEllipseMapping @ SuperEllipseWeights + bias >= 0, 1, -1)


# Dataset previews are separate from module initialization so importing the
# optimization functions has no plotting side effects.
def plot_task1dot2_datasets():
    """Display the original Task 1.2 dataset figures when run directly."""
    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.title("Generated Classification Samples")
    plt.show()

    cubic_score = 2*x1 - 2*x2 + 3*x1**3 + 0.5
    plt.figure()
    plt.scatter(x1, x2, c=CubicLabels, cmap='bwr', alpha=0.5)
    plt.tricontour(
        x1,
        x2,
        cubic_score,
        levels=[0],
        colors="black",
        linewidths=2
    )
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.title("Cubic Classification Dataset")
    plt.grid(True)
    plt.show()

    ellipse_score = x1 + x2 - 0.05*x1**4 - 0.05*x2**4 + 0.5
    plt.figure()
    plt.scatter(x1, x2, c=EllipseLabels, cmap='bwr', alpha=0.5)
    plt.tricontour(
        x1,
        x2,
        ellipse_score,
        levels=[0],
        colors="black",
        linewidths=2
    )
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.title("Super-Ellipse Classification Dataset")
    plt.grid(True)
    plt.show()

def LogRegCostFunction(Phi, p, w, b):
    """
    Compute the cost function for logistic regression.

    Parameters:
    Phi : numpy.ndarray
        The input features mappings of shape (M, N) where M is the number of samples and N is the number of features.
    p : numpy.ndarray
        The true labels of shape (M,) where M is the number of samples.
    w : numpy.ndarray
        The weights of shape (N,) where N is the number of features.
    b : float
        The bias term.

    Returns:
    float
        The computed cost.
        The gradient.
    """
    M = Phi.shape[0]
    scores = Phi @ w + b
    logterm = p * scores 

    # logaddexp evaluates log(1 + exp(-p * score)) without overflow.
    cost = np.sum(np.logaddexp(0, -logterm))

    # The shared scalar term contributes to both parameter derivatives.
    common_term = -p *expit(-logterm)

    grad_w = Phi.T @ common_term
    grad_b = np.sum(common_term)
    return cost, grad_w, grad_b

iter = 2000
stopping_threshold = 1e-6
stepsize = 0.01

b = 0.0

def run_iterations(Phi,p,b,stepsize,iter,stopping_threshold, name=""):
    '''
    Run gradient descent iterations for logistic regression.
    
    Parameters:
    Phi : numpy.ndarray
        The input features mappings of shape (M, N) where M is the number of samples and N is the number of features.
    p : numpy.ndarray
        The true labels of shape (M,) where M is the number of samples.
    b : float
        The bias term.
    stepsize : float
        The learning rate for gradient descent.
    iter : int
        The maximum number of iterations for gradient descent.
    stopping_threshold : float
        The threshold for the gradient norm to determine convergence.
    name : str
        The name of the dataset for plotting purposes.
    '''

    w = np.zeros(Phi.shape[1])
    costs=[]
    iterations=[]
    gradnorms=[]
    for i in range(iter):
        cost, grad_w, grad_b = LogRegCostFunction(Phi, p, w, b)

        costs.append(cost)
        iterations.append(i)
        gradnorm = np.linalg.norm(np.append(grad_w, grad_b))
        gradnorms.append(gradnorm)

        if gradnorm < stopping_threshold:
            print(f"Converged after {i} iterations.")
            break

        w -= stepsize * grad_w
        b -= stepsize * grad_b

    score = Phi @ w + b
    labels_pred = np.where(score >= 0, 1, -1)
    print(f"These are the final results for {name}: \n Final cost: {cost}, iterations: {i}, gradient norm: {gradnorm}")

    misclassified_percentage = 100.0 * float(np.mean(labels_pred != p))
    print(f"Percentage of misclassified points: {misclassified_percentage:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].plot(iterations, costs)
    axes[0].set_title("Cost vs Iterations")

    axes[1].plot(iterations, gradnorms)
    axes[1].set_title("Gradient Norm vs Iterations")

    axes[2].scatter(x1, x2, c=p, cmap='bwr', alpha=0.5)

    if name == "SuperEllipse":

        axes[2].tricontour(x1, x2, score, levels=[0], colors="black",
                    linewidths=2)
    elif name == "Cubic":
        axes[2].tricontour(x1, x2, score, levels=[0], colors="black",
                            linewidths=2)

    axes[2].set_title(f"{name} Classification Dataset with Decision Boundary")

    fig.suptitle(f"{name} Gradient Descent Results", fontsize=16)
    plt.show()
    
    return misclassified_percentage

def run_task1dot2():
    run_iterations(SuperEllipseMapping, EllipseLabels, b, stepsize, iter, stopping_threshold, "SuperEllipse")
    run_iterations(CubicMapping, CubicLabels, b, stepsize, iter, stopping_threshold, "Cubic")

if __name__ == "__main__":
    plot_task1dot2_datasets()
    run_task1dot2()
