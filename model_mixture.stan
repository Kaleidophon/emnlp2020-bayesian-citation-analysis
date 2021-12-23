data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y[N];
  int sizes[J];
  int<lower=2> K;  // Number of mixture components
  vector[K] alpha;  // Dirichlet prior parameters
  
}

parameters {
  matrix<lower=0>[J,K] lambda;
  simplex[K] theta;  // Mixture weights
}

model {
  int pos;
  pos = 1;
  
  for (k in 1:K) {
    // Common Gamma prior for every mixture component
    lambda[1,k] ~ gamma(0.37925113, 0.03268506);
    lambda[2,k] ~ gamma(0.21778377, 0.02369174);
  }
  
  theta ~ dirichlet(alpha); 
  
  for(j in 1:J) {
    for (y_n in segment(y, pos, sizes[j])) {
      real mix_probs[K];  // Prob. of observation under current mix. dist.
      for (k in 1:K) {
         mix_probs[k] = log(theta[k]) + poisson_lpmf(y_n | lambda[j,k]);
      }
      target += log_sum_exp(mix_probs);
    }
    pos = pos + sizes[j];
  }
}

//generated quantities {
//  int ypred[J] = (theta * poisson_rng(lambda[1])) + ((1-theta) * poisson_rng(lambda[1]));
//}
