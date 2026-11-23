// Multivariate Bayesian latent factor model for functional data (any number of modes D).
//
//   eta_n                ~ N_K(beta x_n, I_K)
//   theta_dn | eta_n     ~ N_{p_d}(Lambda_d eta_n, diag(1 / tau_d))
//   y_dn | theta_dn      ~ N(B_dn theta_dn, psi2_d I)            (observed rows B_dn of the basis B_d)
//
// Priors: Lambda_d[j, k] ~ Student-t(nu, 0, omega_k^{-1/2} sqrt(b_lambdavar_d)) with omega_k = delta_1 ... delta_k,
// delta_1 ~ Ga(a1, 1), delta_k ~ Ga(a2, 1) (k >= 2) (MGPS, local precisions integrated out);
// tau_dj ~ Ga(a_tau_d, b_tau_d); psi2_d ~ Inv-Ga(a_psi_d, b_psi_d); beta_kl ~ N(0, sigma_regr_coeffs^2).
//
// The modes are stacked: rows 1..p_1 of Lambda, tau, theta and T belong to mode 1, the next p_2 rows to mode 2, ...
//
// Likelihood (Theorem 2 of the paper, Gram form). For every subject n and mode d with observed basis rows B_dn,
// T_dn are the least-squares coefficients, rss_dn the residual sum of squares and G_dn a square-root factor of
// B_dn' B_dn. Then
//   sum_i (y_dni - (B_dn theta)_i)^2 = rss_dn + || G_dn (T_dn - theta_dn) ||^2,
// so the likelihood, together with the inverse-gamma prior of psi2_d, is
//   InvGamma(psi2_d | a_psi + M_d / 2, b_psi + RSS_d / 2) * exp(-||G (T - theta)||^2 / (2 psi2_d)).
// The sparse matrix A with ||A to_vector(T - theta)||^2 split by mode equal to sum_n ||G_dn (T_dn - theta_dn)||^2 is
// passed in compressed sparse row (CSR) form: its columns follow to_vector(T - theta) (subject by subject), its rows
// are grouped by mode (rows start_d N + 1 .. (start_d + p_d) N belong to mode d).
//
// Coefficients theta_j with a large prior/data precision ratio are sampled non-centred
// (theta_j = mean_j + tau_j^{-1/2} z_j, z_j ~ N(0, 1)); `noncentered` flags them. The sampled block is `theta_tilde`;
// theta itself is rebuilt from it after sampling.

data {
  int<lower=1> N;                    // subjects
  int<lower=1> D;                    // modes
  int<lower=1> K;                    // factors
  int<lower=0> L;                    // covariates
  matrix[L, N] X;                    // covariates (one column per subject)

  int<lower=1> P;                    // sum of the basis sizes
  array[D] int<lower=1> p;           // basis size of every mode
  matrix[P, N] T;                    // least-squares coefficients, modes stacked
  vector<lower=0>[D] sum_rss;        // residual sum of squares of every mode
  array[D] int<lower=0> sum_M;       // number of observed values of every mode
  int<lower=0> n_gram;               // CSR form of A: number of nonzero entries,
  vector[n_gram] gram_w;             //   their values (row by row),
  array[n_gram] int<lower=1, upper=P * N> gram_v;          // their column indices,
  array[P * N + 1] int<lower=1, upper=n_gram + 1> gram_u;  // first entry of every row
  array[P] int<lower=0, upper=1> noncentered;

  vector<lower=0>[D] a_psi;
  vector<lower=0>[D] b_psi;
  vector<lower=0>[D] a_tau;
  vector<lower=0>[D] b_tau;
  real<lower=0> a1;
  real<lower=0> a2;
  vector<lower=0>[D] b_lambdavar;
  real<lower=0> nu;
  real<lower=0> sigma_regr_coeffs;
}

transformed data {
  array[D] int start;                // offset of every mode in the stacked coefficients
  start[1] = 0;
  for (d in 2:D) start[d] = start[d - 1] + p[d - 1];
}

parameters {
  matrix[K, L] regr_coeffs;
  matrix[K, N] eta;
  matrix[P, K] Lambda;
  vector<lower=0>[P] tau;
  matrix[P, N] theta_tilde;
  vector<lower=0>[K] delta;
  vector<lower=0>[D] psi2;
}

transformed parameters {
  vector<lower=0>[K] omega;
  omega[1] = delta[1];
  for (k in 2:K) omega[k] = omega[k - 1] * delta[k];
}

model {
  to_vector(regr_coeffs) ~ normal(0, sigma_regr_coeffs);
  to_vector(eta) ~ normal(to_vector(regr_coeffs * X), 1);

  delta[1] ~ gamma(a1, 1);
  for (k in 2:K) delta[k] ~ gamma(a2, 1);
  for (d in 1:D) {
    segment(tau, start[d] + 1, p[d]) ~ gamma(a_tau[d], b_tau[d]);
    for (k in 1:K)
      segment(col(Lambda, k), start[d] + 1, p[d]) ~ student_t(nu, 0, inv_sqrt(omega[k]) * sqrt(b_lambdavar[d]));
    target += inv_gamma_lpdf(psi2[d] | a_psi[d] + 0.5 * sum_M[d], b_psi[d] + 0.5 * sum_rss[d]);
  }
  {
    matrix[P, N] theta_mean = Lambda * eta;
    matrix[P, N] theta = theta_tilde;
    for (j in 1:P) {
      if (noncentered[j] == 1) {
        theta[j] = theta_mean[j] + inv_sqrt(tau[j]) * theta_tilde[j];
        theta_tilde[j] ~ std_normal();
      } else {
        theta_tilde[j] ~ normal(theta_mean[j], inv_sqrt(tau[j]));
      }
    }
    vector[P * N] r = csr_matrix_times_vector(P * N, P * N, gram_w, gram_v, gram_u, to_vector(T - theta));
    for (d in 1:D) target += -0.5 / psi2[d] * dot_self(segment(r, start[d] * N + 1, p[d] * N));
  }
}
