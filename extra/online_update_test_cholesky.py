import torch

def cholesky_rank1_update(L, x):
    """
    Perform a rank-1 Cholesky update: L ← chol(L @ Lᵀ + x xᵀ)
    L must be lower-triangular Cholesky factor (i.e. L @ L.T = A)
    x is (D,) vector
    """
    x = x.clone()
    D = L.size(0)
    for k in range(D):
        r = torch.sqrt(L[k, k] ** 2 + x[k] ** 2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < D:
            L[k+1:D, k] = (L[k+1:D, k] + s * x[k+1:D]) / c
            x[k+1:D] = c * x[k+1:D] - s * L[k+1:D, k]
    return L


class OnlineBLR:
    r"""Online Bayesian Linear Regression with λ I prior and Cholesky-based updates."""

    def __init__(self, D, noise_var, lambda_reg=1.0, dtype=torch.float64, device="cpu", ns=50000):
        self.D = D
        self.device = torch.device(device)
        self.dtype = dtype

        self.noise_var = torch.as_tensor(noise_var, dtype=dtype, device=self.device)
        self.lambda_reg = lambda_reg

        # Precision Λ₀ = λI  ⇒  Cholesky L₀ = √λ I
        self.L = torch.sqrt(torch.tensor(lambda_reg, dtype=dtype, device=self.device)) \
                 * torch.eye(D, dtype=dtype, device=self.device)           # (D,D) lower-tri
        self.mu = torch.zeros((D, 1), dtype=dtype, device=self.device)     # (D,1)
        self.ns=ns
        self.z = torch.randn(self.ns, self.D, dtype=self.dtype, device=self.device)   # (n,D)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _precision_solve(self, B):
        """Solve Λ X = B using stored Cholesky factor L (Λ = LLᵀ)."""
        return torch.cholesky_solve(B, self.L, upper=False)

    @torch.no_grad()
    def update(self, Phi_new, y_new):
        Φ = Phi_new.to(self.device, self.dtype).reshape(-1, self.D)
        y = y_new.to(self.device, self.dtype).reshape(-1, 1)

        # Accumulate sufficient statistics
        if not hasattr(self, "PhiTPhi"):
            self.PhiTPhi = torch.zeros((self.D, self.D), dtype=self.dtype, device=self.device)
            self.PhiTy = torch.zeros((self.D, 1), dtype=self.dtype, device=self.device)

        self.PhiTPhi += Φ.T @ Φ
        self.PhiTy += Φ.T @ y

        # Posterior precision: Λ = λI + ΦᵀΦ / σ²
        Λ = self.lambda_reg * torch.eye(self.D, dtype=self.dtype, device=self.device) + self.PhiTPhi / self.noise_var
        self.L = torch.linalg.cholesky(Λ)

        # Solve Λ μ = Φᵀy / σ²
        rhs = self.PhiTy / self.noise_var
        self.mu = self._precision_solve(rhs)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, Phi_star, return_std=False):
        Φs = Phi_star.to(self.device, self.dtype).reshape(-1, self.D)  # (M,D)
        mean = Φs @ self.mu                                           # (M,1)

        if not return_std:
            return mean

        # var = σ² + diag(Φs Σ Φsᵀ) ;  Σ = Λ⁻¹  via triangular solves
        # Compute q = L⁻¹ Φsᵀ  →  ΣΦsᵀ = L⁻ᵀ q
        q = torch.linalg.solve_triangular(self.L, Φs.T, upper=False)      # (D,M)
        var = (q ** 2).sum(dim=0, keepdim=True).T + self.noise_var        # (M,1)
        return mean, torch.sqrt(var)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_weights(self, n=1):
        """Draw n samples w ∼ 𝒩(μ, Σ)."""
        # self.z = torch.randn(self.ns, self.D, dtype=self.dtype, device=self.device) 
        # Σ½  =  L⁻ᵀ
        samples = torch.linalg.solve_triangular(
            self.L, self.z.T, upper=False)    
        return self.mu.squeeze(-1) + samples.T                   # (n, D)
        # L_inv_T = torch.linalg.solve_triangular(self.L, torch.eye(self.D, device=self.device,
        #                                                            dtype=self.dtype),
        #                                         upper=False).T
        # return (self.mu.squeeze() + self.z @ L_inv_T).to(self.dtype)            # (n,D)
        # return self.mu.squeeze() + self.z @ self.L.T    # (n, D)

import timeit

torch.manual_seed(0)
now = timeit.default_timer()

num_features = 500  # Number of features (dimensions)

for i in range(1):
    # ---------- Synthetic data ----------
    N = 10
    true_w = torch.full((num_features,), 2.0, dtype=torch.float64)   # e.g., w = [2, 2, ..., 2]
    noise_std = 0.2
    x = torch.rand(N, num_features, dtype=torch.float64)
    noise = noise_std * torch.randn(N, 1, dtype=torch.float64)
    y = x @ true_w.view(-1, 1) + noise                    # shape: (N, 1)
    print("x:\n", x, "\ny:\n", y)

    noise_var = noise_std ** 2
    lambda_reg = 1e-1  # Regularization parameter

    # ---------- Batch posterior ----------
    batch_blr = OnlineBLR(D=num_features, noise_var=noise_var, lambda_reg=lambda_reg,ns=0)
    batch_blr.update(x, y)
    mu_batch = batch_blr.mu
    prec_batch = batch_blr.L

    # ---------- Online posterior ----------
    online_blr = OnlineBLR(D=num_features, noise_var=noise_var, lambda_reg=lambda_reg, ns=0)
    print("Step | Online μ      | Online Λ")
    for i in range(N):
        online_blr.update(x[i], y[i])
        print(f"{i+1:>4} | {torch.mean(online_blr.mu):>12.6f} | {torch.mean(online_blr.L):>9.2f}")

    # ---------- Comparison ----------
    print("-" * 38)
    print(f"Batch| {torch.mean(mu_batch):>12.6f} | {torch.mean(prec_batch):>9.2f}")


    # ---------- Both offline and online ----------

    batch_online_blr = OnlineBLR(D=num_features, noise_var=noise_var, lambda_reg=lambda_reg, ns=100000)
    batch_online_blr.update(x[:5], y[:5])
    mu_batch = batch_online_blr.mu
    prec_batch = batch_online_blr.L
    print("Step | Online μ      | Online Λ")
    for i in range(5,N):
        batch_online_blr.update(x[i], y[i])
        print(f"{i+1:>4} | {torch.mean(batch_online_blr.mu):>12.6f} | {torch.mean(batch_online_blr.L):>9.2f}")

    now_1 = timeit.default_timer()
    print(batch_online_blr.sample_weights(10000))  # Sample 5 weight vectors
    now_2 = timeit.default_timer()
    print(f"Time for sampling: {now_2 - now_1:.4f} seconds")

end_time = timeit.default_timer()
print(f"Time taken: {end_time - now:.4f} seconds")















    # ------------------------------------------------------------------ #
    # @torch.no_grad()
    # def update(self, Phi_new, y_new):
    #     r"""Add mini-batch (Φ, y) and update posterior.

    #     Cheap form: rank-k update to precision using Cholesky downdate+update.
    #     """
    #     Φ = Phi_new.to(self.device, self.dtype).reshape(-1, self.D)   # (N,D)
    #     y = y_new.to(self.device, self.dtype).reshape(-1, 1)          # (N,1)
    #     N = Φ.shape[0]

    #     # ---------- Update precision Λ  (rank-k update) ---------------
    #     # # Λ_new = Λ + ΦᵀΦ / σ²
    #     S = Φ / torch.sqrt(self.noise_var)        # (N,D)  scaled design
    #     # # For each row sᵢ of S, do Λ ← Λ + sᵢᵀ sᵢ  (rank-1)
    #     # for s in S:
    #     #     self.L = torch.linalg.cholesky_update(self.L, s.unsqueeze(1), True)  # lower=True

    #     # Rank-1 updates via SciPy
    #     for i in range(S.shape[0]):
    #         self.L = cholesky_rank1_update(self.L, S[i])

    #     # ---------- Update mean μ -------------------------------------
    #     # RHS_new = Λμ + Φᵀy / σ²
    #     rhs_increment = (Φ.T @ y) / self.noise_var                        # (D,1)
    #     rhs = self.L @ (self.L.T @ self.mu) + rhs_increment               # Λ μ + …
    #     # Solve Λ μ_new = rhs   →   μ_new
    #     self.mu = self._precision_solve(rhs)