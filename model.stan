data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y[N];
  int sizes[J];
  
}

parameters {
  matrix<lower=0>[J,2] lambda;
  real<lower=0, upper=1> theta;
}

model {
  int pos;
  pos = 1;
  lambda[,1] ~ gamma(0.37925113, 0.03268506);
  lambda[,2] ~ gamma(0.21778377, 0.02369174);
  
  theta ~ beta(0.5, 0.5);

  for(j in 1:J) {
    target += log_mix(theta,
                      poisson_lpmf(segment(y, pos, sizes[j]) | lambda[j,1]),
                      poisson_lpmf(segment(y, pos, sizes[j]) | lambda[j,2]));
    
    pos = pos + sizes[j];
  }
}

//generated quantities {
//  int ypred[J] = (theta * poisson_rng(lambda[1])) + ((1-theta) * poisson_rng(lambda[1]));
//}
