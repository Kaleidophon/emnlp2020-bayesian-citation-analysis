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
  vector[J] c;
 }
 
model {
  int pos;
  pos = 1;
  
  lambda[1] ~ gamma(0.37925113, 0.03268506); // Main 
  lambda[2] ~ gamma(0.21778377, 0.02369174); // Findings
  
  // Set a lower limit for the slope parameter s.t. the rate parameter never 
  // becomes negative.
  c[1] ~ normal(1.778491, 18.216528) T[-lambda[1],]; // Main
  c[2] ~ normal(1.72022, 24.08663) T[-lambda[2],]; // Findings
  
  for(j in 1:J) {
    rho[j] ~ beta(2,2);
    segment(y2, pos, sizes[j]) ~ bernoulli(rho[j]);
    
    // Add log probabilities for both features
    target += bernoulli_lpmf(segment(y2, pos, sizes[j]) | rho[j]);

    // We have to introduce a loop here because every observation has its own
    // modified rate parameter lambda depending on its length
    for (n in 1:sizes[j]) {
      real new_lambda;
      
      new_lambda = lambda[j] + c[j] * segment(y2, pos, sizes[j])[n];
      target += poisson_lpmf(segment(y1, pos, sizes[j])[n] | new_lambda);
    }
    
    pos = pos + sizes[j];
  }
}

generated quantities {
vector[J] ypred;
  for(j in 1:J) {
    for (n in 1:N) {
      real new_lambda = lambda[j] + c[j] * bernoulli_rng(beta_rng(2, 2));
      ypred[j] = poisson_rng(new_lambda);
    }
  }
}
