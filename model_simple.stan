data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> y[N];
  int sizes[J];
  
}

 parameters {
   vector<lower=0>[J] lambda;
 }
 
model {
  int pos;
  pos = 1;
  for(j in 1:J) {
    segment(y, pos, sizes[j]) ~ poisson(lambda[j]);
    pos = pos + sizes[j];
  }
}

generated quantities {
  int ypred[J] = poisson_rng(lambda);
}