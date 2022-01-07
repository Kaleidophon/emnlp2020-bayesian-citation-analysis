data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y1[N];
  int<lower=0> y2[N];
  vector<lower=0>[N] is_long;
  int sizes[J];
  int<lower=0> draws;
}

 parameters {
  vector<lower=0>[J] lambda;
  vector<lower=0, upper=1>[J] rho;
  vector<lower=0>[J] c;
 }
 
model {
  int pos;
  pos = 1;
  
  lambda[1] ~ gamma(0.37925113, 0.03268506); // Main 
  lambda[2] ~ gamma(0.21778377, 0.02369174); // Findings

  // truncate
  c[1] ~ normal(1.778491, 18.216528) T[-lambda[1],]; // Main
  c[2] ~ normal(1.72022, 24.08663) T[-lambda[2],]; // Findings
  
  rho ~ beta(2,2);
  for(j in 1:J) {
    segment(y1, pos, sizes[j]) ~ poisson(lambda[j] + (c[j] * segment(is_long, pos, sizes[j])));
    segment(y2, pos, sizes[j]) ~ bernoulli(rho[j]);
    
    pos = pos + sizes[j];
  }
}

generated quantities {
  vector[N] log_lik;
  vector[draws] findings_ypred;
  vector[draws] main_ypred;
  
  // Get predictive posterior draws
  for (i in 1:draws) {
    findings_ypred[i] = poisson_rng(lambda[2] + c[2] * bernoulli_rng(rho[2]));
    main_ypred[i] = poisson_rng(lambda[1] + c[1] * bernoulli_rng(rho[1]));
  }
  
  // Score log-likelihood for observations
  for (i in 1:N) {
    if (i < sizes[1]) {
      log_lik[i] = poisson_lpmf(y1[i] | lambda[1] + c[1] * is_long[1]);
    }
    else {
      log_lik[i] = poisson_lpmf(y1[i] | lambda[2] + c[2] * is_long[2]);
    }
  }
}


