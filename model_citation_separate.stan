data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y1[N];
  int<lower=0, upper=1> y2[N];
  int sizes[J];
  
}

 parameters {
  vector<lower=0>[J] lambda;
  vector<lower=0, upper=1>[J] rho;
  vector[J] mu; // Slope prior mean
  vector[J] sigma; // Slope prior std
  vector[J] c;
 }
 
model {
  int pos;
  matrix[J, 2] lambda_adj;
  pos = 1;
  
  lambda_adj[1, 2] = 1;
  lambda_adj[2, 2] = 1;
  lambda[1] ~ gamma(0.37925113, 0.03268506); // Main 
  lambda[2] ~ gamma(0.21778377, 0.02369174); // Findings
  c[1] ~ normal(1.778491, 18.216528); // Main
  c[2] ~ normal(1.72022, 24.08663); // Findings
  
  for(j in 1:J) {
    // TODO: Vectorize this to make this faster?
    for (n in pos:pos+sizes[j]-1) {
      rho[j] ~ beta(0.5, 0.5); // TODO: Also try this with beta(2, 2)
      y2[n] ~ bernoulli(rho[j]);
      
      // Add log probabilities for both features
      target += bernoulli_lpmf(y2[n] | rho[j]);
      
      // Super awkward way to make sure rate parameter is at least 1
      lambda_adj[j, 1] = lambda[j] + c[j] * y2[n];
      target += poisson_lpmf(y1[n] | max(lambda_adj[j]));
    }
    //segment(y1, pos, sizes[j]) ~ poisson(lambda[j] + c[j] * y2);
    pos = pos + sizes[j];
  }
}

// TODO: Figure this out
//generated quantities {
//  int ypred[J] = poisson_rng(lambda);
//}
