import torch

class OnlineBLR:
    def __init__(self, D, noise_var, lambda_reg=1.0, dtype=torch.float64, device="cpu"):
        self.D = D
        self.device = torch.device(device)
        self.dtype = dtype
        self.noise_var = torch.tensor(noise_var, dtype=dtype, device=device)
        self.lambda_reg = lambda_reg
        self.inv_Sigma = lambda_reg * torch.eye(D, dtype=dtype, device=device)  # Prior precision: λI
        self.mu = torch.zeros((D, 1), dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, Phi_new, y_new):
        Phi_new = Phi_new.to(self.device).reshape(-1, self.D).to(self.dtype)
        y_new = y_new.to(self.device).reshape(-1, 1).to(self.dtype)

        PhiT = Phi_new.T
        A = PhiT @ Phi_new
        b = PhiT @ y_new

        Sigma_post_inv = self.inv_Sigma + A / self.noise_var
        rhs = self.inv_Sigma @ self.mu + b / self.noise_var
        mu_post = torch.linalg.solve(Sigma_post_inv, rhs)

        self.inv_Sigma = Sigma_post_inv
        self.mu = mu_post

    @torch.no_grad()
    def predict(self, Phi_star, return_std=False):
        Phi_star = Phi_star.to(self.device).reshape(-1, self.D).to(self.dtype)
        mean = Phi_star @ self.mu

        if not return_std:
            return mean

        # Predictive variance: σ² + Φ* Λ⁻¹ Φ*ᵀ
        sol = torch.linalg.solve(self.inv_Sigma, Phi_star.T)
        var = (Phi_star * sol.T).sum(dim=1, keepdim=True) + self.noise_var
        std = torch.sqrt(var)
        return mean, std
    
    @torch.no_grad()
    def sample_weights(self, n: int = 1) -> torch.Tensor:
        """
        Efficiently sample n weights from the posterior w ∼ 𝒩(μ, Σ), with Σ = Λ⁻¹,
        without explicitly computing Σ.

        Returns:
            samples: (n, D) tensor where each row is one weight sample
        """
        # Cholesky factor of Λ = inv_Sigma
        L = torch.linalg.cholesky(self.inv_Sigma)              # (D, D)

        # Solve Lᵀ x = εᵀ ⇒ xᵀ = L⁻ᵀ εᵀ ⇒ x = (L⁻ᵀ εᵀ)ᵀ = ε @ L⁻¹
        eps = torch.randn(n, self.D, dtype=self.dtype, device=self.device)  # (n, D)
        samples = torch.linalg.solve_triangular(
            L, eps.T, upper=False
        )                                                    # (n, D)

        return self.mu.squeeze(-1) + samples.T                   # (n, D)
    
    @torch.no_grad()
    def sample_weights_old(self, n=1):
        """
        Sample n weight vectors from the posterior w ∼ 𝒩(μ, Σ)
        Returns: (n, D) tensor where each row is one sampled weight vector.
        """
        Sigma = torch.linalg.inv(self.inv_Sigma)  # (D, D)
        L = torch.linalg.cholesky(Sigma)          # (D, D), lower-triangular
        eps = torch.randn(n, self.D, dtype=self.dtype, device=self.device)  # (n, D)
        samples = self.mu.squeeze() + eps @ L.T    # (n, D)
        return samples
    
    @torch.no_grad()
    def sample_weights_mnv(self, n: int = 1) -> torch.Tensor:
        """
        Draw `n` weight samples w ~ 𝒩(μ, Σ)   with Σ = Λ⁻¹.

        Returns
        -------
        samples : (n, D) tensor
        """
        # Posterior covariance
        Sigma = torch.linalg.inv(self.inv_Sigma)          # (D, D)

        # Multivariate normal sampler
        mvn = torch.distributions.MultivariateNormal(
            loc=self.mu.squeeze(-1),                      # (D,)
            covariance_matrix=Sigma
        )
        return mvn.sample((n,))  
    
import timeit


now = timeit.default_timer()

num_features = 1000  # Number of features (dimensions)

for iter in range(2):
    torch.manual_seed(0)
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
    batch_blr = OnlineBLR(D=num_features, noise_var=noise_var, lambda_reg=lambda_reg)
    batch_blr.update(x, y)
    mu_batch = batch_blr.mu
    prec_batch = batch_blr.inv_Sigma

    # ---------- Online posterior ----------
    online_blr = OnlineBLR(D=num_features, noise_var=noise_var, lambda_reg=lambda_reg)
    print("Step | Online μ      | Online Λ")
    for i in range(N):
        online_blr.update(x[i], y[i])
        print(f"{i+1:>4} | {torch.mean(online_blr.mu):>12.6f} | {torch.mean(online_blr.inv_Sigma):>9.2f}")

    # ---------- Comparison ----------
    print("-" * 38)
    print(f"Batch| {torch.mean(mu_batch):>12.6f} | {torch.mean(prec_batch):>9.2f}")


    # ---------- Both offline and online ----------

    batch_online_blr = OnlineBLR(D=num_features, noise_var=noise_var, lambda_reg=lambda_reg)
    batch_online_blr.update(x[:5], y[:5])
    mu_batch = batch_online_blr.mu
    prec_batch = batch_online_blr.inv_Sigma
    print("Step | Online μ      | Online Λ")
    for i in range(5,N):
        batch_online_blr.update(x[i], y[i])
        print(f"{i+1:>4} | {torch.mean(batch_online_blr.mu):>12.6f} | {torch.mean(batch_online_blr.inv_Sigma):>9.2f}")

    if iter==0:
        # time these two
        now_1 = timeit.default_timer()
        samples_old = batch_online_blr.sample_weights_old(50)  # Sample 5 weight vectors
        print(samples_old)
        now_2 = timeit.default_timer()
        print(f"Sampling time: {now_2 - now_1:.4f} seconds")
    else:
        now_1_eff = timeit.default_timer()
        samples_efficient = batch_online_blr.sample_weights(50)  # Sample 5 weight vectors
        print(samples_efficient)
        now_2_eff = timeit.default_timer()
        print(f"Sampling time (old): {now_2_eff - now_1_eff:.4f} seconds")

mu1 = samples_old.mean(dim=0)
mu2 = samples_efficient.mean(dim=0)
cov1 = torch.cov(samples_old.T)
cov2 = torch.cov(samples_efficient.T)
print("Mean difference:", torch.norm(mu1 - mu2))
print("Covariance difference:", torch.norm(cov1 - cov2))
    # now_11 = timeit.default_timer()
    # print(batch_online_blr.sample_weights_mnv(500000))  # Sample 5 weight vectors
    # now_21 = timeit.default_timer()
    
    # print(f"Sampling time (mvn): {now_21 - now_11:.4f} seconds")


end_time = timeit.default_timer()
print(f"Time taken: {end_time - now:.4f} seconds")