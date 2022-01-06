data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y1[N];
  int<lower=0> y2[N];
  vector<lower=0>[N] is_long;
  int sizes[J];
}

 parameters {
  vector<lower=0>[J] lambda;
  vector<lower=0, upper=1>[J] rho;
  vector<lower=0>[J] c;
 }
 
model {
  int pos;
  pos = 1;
  
  rho ~ beta(2,2);
  
  for (j in 1:J) {
    lambda[j] ~ gamma(0.29527993, 0.02758679);
    c[j] ~ normal(1.779526, 21.512018) T[-lambda[j],];
  }
  
  for(j in 1:J) {
    segment(y1, pos, sizes[j]) ~ poisson(lambda[j] + (c[j] * segment(is_long, pos, sizes[j])));
    segment(y2, pos, sizes[j]) ~ bernoulli(rho[j]);
    
    pos = pos + sizes[j];
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] findings_ypred;
  vector[N] main_ypred;
  
  for (i in 1:N) {
    findings_ypred[i] = poisson_rng(lambda[2] + c[2] * bernoulli_rng(rho[2]));
    main_ypred[i] = poisson_rng(lambda[1] + c[1] * bernoulli_rng(rho[1]));
    
    if (i < sizes[1]) {
      log_lik[i] = poisson_lpmf(y1[i] | lambda[1] + c[1] * is_long[1]);
    }
    else {
      log_lik[i] = poisson_lpmf(y1[i] | lambda[2] + c[2] * is_long[2]);
    }
  }
}

