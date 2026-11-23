// Marginal likelihood of the MvtBLF data model, for maximum-likelihood estimation (no priors; optimise with
// jacobian = false). The scores eta_n and the basis coefficients theta_dn are integrated out:
//
//   eta_n = beta x_n + z_n,  z_n ~ N(0, I_K);   theta_dn = Lambda_d eta_n + v_dn,  v_dn ~ N(0, diag(sigma2_d));
//   y_dn = B_dn theta_dn + e_dn,  e_dn ~ N(0, psi2_d I).
//
// Per subject and mode, R (p_d x p_d, R'R = B_dn'B_dn), u = R T_dn, rss and m (observed values) are sufficient:
//   log p(y_dn | eta_n) = -m/2 log(2 pi psi2) + p/2 log psi2 - rss/(2 psi2) - 1/2 log|E| - 1/2 a'a,
//   E = psi2 I + R diag(sigma2) R',  a = chol(E)^{-1} (u - R Lambda_d eta_n).
// Integrating z_n gives, with G_d = chol(E)^{-1} R Lambda_d, H = I + sum_d G_d'G_d and h = sum_d G_d'a_d (a_d at eta_n = beta x_n):
//   log p(y_n) = sum_d c_d - 1/2 log|H| - 1/2 (sum_d a_d'a_d - h'H^{-1}h).

functions {
  real subject_loglik(matrix R, vector u, vector rss, vector m, matrix Lambda, vector sigma2, vector psi2,
                      vector mean_eta, array[] int p) {
    int K = cols(Lambda);
    int D = size(p);
    matrix[K, K] H = identity_matrix(K);
    vector[K] h = rep_vector(0, K);
    real q = 0;
    real c = 0;
    int s = 0;
    for (d in 1:D) {
      int pd = p[d];
      matrix[pd, pd] Rd = block(R, s + 1, s + 1, pd, pd);
      matrix[pd, K] RL = Rd * Lambda[(s + 1):(s + pd)];
      matrix[pd, pd] LE = cholesky_decompose(add_diag(diag_post_multiply(Rd, sigma2[(s + 1):(s + pd)]) * Rd', psi2[d]));
      vector[pd] a = mdivide_left_tri_low(LE, u[(s + 1):(s + pd)] - RL * mean_eta);
      matrix[pd, K] G = mdivide_left_tri_low(LE, RL);
      q += dot_self(a);
      h += G' * a;
      H += crossprod(G);
      c += -0.5 * m[d] * (log(2 * pi()) + log(psi2[d])) + 0.5 * pd * log(psi2[d]) - 0.5 * rss[d] / psi2[d]
           - sum(log(diagonal(LE)));
      s += pd;
    }
    matrix[K, K] LH = cholesky_decompose(H);
    vector[K] w = mdivide_left_tri_low(LH, h);
    return c - sum(log(diagonal(LH))) - 0.5 * (q - dot_self(w));
  }
}

data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> K;
  int<lower=0> L;
  matrix[L, N] X;
  int<lower=1> P;
  array[D] int<lower=1> p;
  array[N] matrix[P, P] R;           // blkdiag over modes of the square-root factors of every subject
  matrix[P, N] U;                    // R_n T_n, modes stacked
  matrix[D, N] rss;                  // residual sums of squares
  matrix[D, N] n_obs;                // observed values
  real<lower=0> variance_floor;      // lower bound of every variance
}

parameters {
  matrix[P, K] Lambda;
  vector<lower=variance_floor>[P] sigma2;
  vector<lower=variance_floor>[D] psi2;
  matrix[K, L] beta;
}

model {
  for (n in 1:N)
    target += subject_loglik(R[n], U[, n], rss[, n], n_obs[, n], Lambda, sigma2, psi2, beta * X[, n], p);
}
