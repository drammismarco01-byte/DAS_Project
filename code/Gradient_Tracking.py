import numpy as np


class GradientTracking:

    def __init__(
        self, z_init, b_points, Q_matrices, adj_matrix, alpha, tol=1e-5
    ):
        """Initializes the Gradient Tracking algorithm for a network of N

        agents.

        Parameters:
        - z_init: np.ndarray (N, d) -> Initial positions/estimates z_i^0 for
        each agent
        - b_points: np.ndarray (N, d) -> Target points b_i for the
        minimum-centered canonical form
        - Q_matrices: np.ndarray (N, d, d) -> Curvature matrices Q_i for each
        agent
        - adj_matrix: np.ndarray (N, N) -> Adjacency matrix of the communication
        graph
        - alpha: float -> Step-size (learning rate)
        - tol: float -> Tolerance threshold for stopping criteria
        """
        self.N = z_init.shape[0]
        self.d = z_init.shape[1]
        self.alpha = alpha
        self.tol = tol  # <--- NUOVO ATTRIBUTO: Tolleranza per la convergenza

        self.b = np.array(b_points, dtype=float)
        self.Q = np.array(Q_matrices, dtype=float)

        # 1. Computation of the doubly stochastic weight matrix W via Metropolis-Hastings
        self.W = self._compute_metropolis_hastings(adj_matrix)

        # 2. Initialization of the current position X (N x d) with initial z_i values
        self.X = np.array(z_init, dtype=float)

        # 3. Initialization of the tracking variable Y (N x d): s_i^0 = grad_i(z_i^0)
        self.Y = np.zeros((self.N, self.d))
        for i in range(self.N):
            self.Y[i] = self.compute_local_gradient(i, self.X[i])

    @staticmethod
    def _compute_metropolis_hastings(adj_matrix):
        """Computes the doubly stochastic weight matrix W from the adjacency

        matrix.
        """
        n = adj_matrix.shape[0]
        W = np.zeros((n, n))
        degrees = np.sum(adj_matrix, axis=1)

        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] == 1:
                    W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))

        for i in range(n):
            W[i, i] = 1.0 - np.sum(W[i, :])
        return W

    def compute_local_cost(self, agent_idx, z):
        """Computes the local cost function l_i(z) in canonical form centered at

        the minimum:

        l_i(z) = 0.5 * (z - b_i)^T * Q_i * (z - b_i)
        """
        diff = z - self.b[agent_idx]
        return 0.5 * float(diff.T @ self.Q[agent_idx] @ diff)

    def compute_local_gradient(self, agent_idx, z):
        """Computes the local gradient of the cost function:

        grad_i(z) = Q_i * (z - b_i)
        """
        diff = z - self.b[agent_idx]
        return self.Q[agent_idx] @ diff

    def step(self):
        """Executes an update step (k -> k+1) according to the Gradient Tracking

        algorithm:

        1. z^{k+1} = W * z^k - alpha * s^k
        2. s^{k+1} = W * s^k + grad(z^{k+1}) - grad(z^k)
        """
        # Step 1: Position estimate update
        X_next = self.W @ self.X - self.alpha * self.Y

        # Step 2: Computation of old and new local gradients
        grad_curr = np.zeros_like(self.X)
        grad_next = np.zeros_like(self.X)
        for i in range(self.N):
            grad_curr[i] = self.compute_local_gradient(i, self.X[i])
            grad_next[i] = self.compute_local_gradient(i, X_next[i])

        # Step 3: Gradient tracking update
        Y_next = self.W @ self.Y + grad_next - grad_curr

        # Internal class state update
        self.X = X_next
        self.Y = Y_next

    def get_global_cost(self):
        """Computes the global cost function sum_i l_i(z_bar) evaluated at the

        current average estimate (consensus).
        """
        z_bar = np.mean(self.X, axis=0)
        return sum(self.compute_local_cost(i, z_bar) for i in range(self.N))

    def get_global_gradient_norm(self):
        """Computes the gradient norm of the global cost function ||sum_i

        grad_i(z_bar)|| evaluated at the current average estimate.
        """
        z_bar = np.mean(self.X, axis=0)
        total_grad = sum(
            self.compute_local_gradient(i, z_bar) for i in range(self.N)
        )
        return float(np.linalg.norm(total_grad))

    @property
    def can_stop(self) -> bool:
        """Determines whether the algorithm has converged and can stop.

        Checks two conditions:
        1. Consensus Error: All agents are sufficiently close to their mean.
        2. Optimality: The norm of the global gradient at consensus is below tolerance.
        """
        z_bar = np.mean(self.X, axis=0)

        # Maximum distance of any agent from the consensus value
        consensus_error = float(np.max(np.linalg.norm(self.X - z_bar, axis=1)))

        # Norm of the sum of local gradients at consensus point
        grad_norm = self.get_global_gradient_norm()

        return consensus_error < self.tol and grad_norm < self.tol