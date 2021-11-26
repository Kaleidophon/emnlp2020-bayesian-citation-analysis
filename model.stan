data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y[N, J];
}

parameters {
  vector<lower=0>[J] lambda;
}

model {
  for(j in 1:J) {
    y[, j] ~ poisson(lambda[j]);
  }
}
