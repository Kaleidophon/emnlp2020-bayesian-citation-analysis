data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y[N];
  int sizes[J];
  int<lower=2> K;  // Number of mixture components
  vector[K] alpha[J];  // Dirichlet prior parameters
  
}

parameters {
  matrix<lower=0>[J,K] lambda; // Rate
  simplex[K] theta[J];  // Mixture weights, one simplex per main / findings
}

model {
  int pos;
  pos = 1;
  
  for (k in 1:K) {
    // Common Gamma prior for every mixture component
    if (k == 1) {
      lambda[1,k] ~ gamma(0.3577605, 0.5981308); // Main
      lambda[2,k] ~ gamma(0.2341311, 0.4838710); // Findings
    }
    else {
      lambda[1,k] ~ gamma(0.60074470, 0.04131175); // Main
      lambda[2,k] ~ gamma(0.59930843, 0.03945434); // Findings
    }
  }
  
  for(j in 1:J) {
    theta[j] ~ dirichlet(alpha[j]); // Draw mixture components from dirichlet
    
    for (y_n in segment(y, pos, sizes[j])) {
      real mix_probs[K];  // Prob. of observation under current mix. dist.
      for (k in 1:K) {
         mix_probs[k] = log(theta[j,k]) + poisson_lpmf(y_n | lambda[j,k]);
      }
      target += log_sum_exp(mix_probs); // Log of sum of mix. probs.
    }
    pos = pos + sizes[j];
  }
}

generated quantities {
vector[J] log_lik;
  for(j in 1:J) {
      real mix_probs[K];  // Prob. of observation under current mix. dist.
      for (k in 1:K) {
         mix_probs[k] = log(theta[j,k]) + poisson_lpmf(y[k] | lambda[j,k]);
      }
      log_lik[j] = log_sum_exp(mix_probs);
    }
}
