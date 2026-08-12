import numpy as np
from scipy.special import expit

class GradientTrackingLogReg:

    def __init__(
        self, z_init, feature_subsets, label_subsets, adj_matrix, alpha, tol=1e-5
    ):
        """Initializes the Gradient Tracking algorithm for a network of N

        agents.

        Parameters:
        - z_init: np.ndarray (N, d) -> Initial positions/estimates z_i^0 for
        each agent
        - feature_subsets: list of np.ndarray (N, d) -> Input features for each agent
        - label_subsets: list of np.ndarray (N,) -> True labels for each agent
        - adj_matrix: np.ndarray (N, N) -> Adjacency matrix of the communication
        graph
        - alpha: float -> Step-size (learning rate)
        - tol: float -> Tolerance threshold for stopping criteria
        """
        # Network and classifier dimensions are determined by the initial state.
        self.N = z_init.shape[0]
        self.d = z_init.shape[1]

        self.alpha = alpha
        self.tol = tol

        # Local datasets remain separate because each agent evaluates its own loss.
        self.local_features = []
        self.local_labels = []

        for agent in range(self.N):
            agent_features = np.asarray(
                feature_subsets[agent],
                dtype=float
            )

            agent_labels = np.asarray(
                label_subsets[agent],
                dtype=float
            )

            # A constant final column represents the classifier bias.
            agent_features_with_bias = np.column_stack((
                agent_features,
                np.ones(len(agent_features))
            ))

            self.local_features.append(agent_features_with_bias)
            self.local_labels.append(agent_labels)


        # Metropolis-Hastings weights preserve averages during communication.
        self.W = self._compute_metropolis_hastings(adj_matrix)

        # X stores agent estimates; Y tracks their changing local gradients.
        self.X = np.array(z_init, dtype=float)
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

    def compute_local_cost(self, agent_idx, parameters):
        """Compute one agent's summed logistic loss."""
        p = self.local_labels[agent_idx]
        scores = self.local_features[agent_idx] @ parameters
        logterm = p * scores

        cost = np.sum(np.logaddexp(0, -logterm))
        return cost

    def compute_local_gradient(self, agent_idx, parameters):
        """Compute the gradient of one agent's logistic loss."""
        p = self.local_labels[agent_idx]
        phi = self.local_features[agent_idx]

        # The feature matrix already contains its final bias column, so the
        # bias derivative is the final component of the matrix product.
        scores = phi @ parameters
        logterm = p * scores

        common_term = -p * expit(-logterm)

        gradient = phi.T @ common_term
        return gradient

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
