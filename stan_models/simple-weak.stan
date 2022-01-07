data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y[N];
  int sizes[J];
  int<lower=0> draws;
}

 parameters {
   vector<lower=0>[J] lambda;
 }
 
model {
  int pos;
  pos = 1;
  
  lambda[1] ~ gamma(0.3577605, 0.5981308); // Main
  lambda[2] ~ gamma(0.2341311, 0.4838710); // Findings
  
  for(j in 1:J) {
    segment(y, pos, sizes[j]) ~ poisson(lambda[j]);
    pos = pos + sizes[j];
  }
}

generated quantities {
  vector[N] log_lik;
  vector[draws] findings_ypred;
  vector[draws] main_ypred;
  
  // Get predictive posterior draws
  for (i in 1:draws) {
    main_ypred[i] = poisson_rng(lambda[1]);
    findings_ypred[i] = poisson_rng(lambda[2]);
  }
  
  // Score log-likelihood for observations
  for (i in 1:N) {
    if (i < sizes[1]) {
      log_lik[i] = poisson_lpmf(y[i] | lambda[1]);
    }
    else {
      log_lik[i] = poisson_lpmf(y[i] | lambda[2]);
    }
  }
}
