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
  
  for (j in 1:J) {
    lambda[j] ~ gamma(0.29527993, 0.02758679);
    c[j] ~ normal(1.779526, 21.512018) T[-lambda[j],];
  }
  
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
