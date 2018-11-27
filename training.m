%Minor Modifications to Cart Pole


data_handle =  "data_all.mat";
opt_method = 'CG';
load(data_handle);
%Change this into a better tool, idea is to allow user to specficy the
%order of optimization of hyperparameters by passing a vector of strings.
%Should give a default, currently ignored, also add length parameters
%saying how many iterations in each set of iterations
params_to_optimize = ['hyp','qx','all'];


N = size(train, 2);     % number of time series
D = size(train(1).y,2); % observation dimension
%Currently assumed to match observation dimension
E = D;                  % latent state dimension
U = size(train(1).u,2); % action dimension
F = E + U;
M = 100;                 % number of inducing inputs

%intialize the kernel hyperparameters, do kernels have a fixed variance?
for e = 1:E
  hyp(e).l(1:D,1) = 0;
  hyp(e).l(D+1:F,1) = log(std(train(1).u, 1));
  hyp(e).pn = 0;
end
%intitialise latent means to match observed data. This assumes the latent
%data and the observations are of the same dimenesion. Would sampling a
%random projection ExF matrix work here to address the case when E>F? The
%latent states are rescaled so as to have variance 1. It looks like the
%inducing variables end up sort of normal, though this could just be that
%they are intialized this way. This would cut down the number of
%variational parameters by a factor of M, but we can probably only use 2 per dimension on cart pole if
%add GPU support?
[qx(1:N).m] = deal(train(1:N).y);

s = std(qx(1).m);
for n = 1:N
  
  qx(n).m = (qx(n).m ./ s)';
  T = size(train(n).y, 1);
  S = 0.1*repmat(eye(2*E),ceil(T/2),1);
  %initialize the latent state covariance matrix to an identity matrix
  qx(n).s = reshape(S(1:E*T,:)',2*E,E,T);  
end
%initialize inducing features to be Gaussian distributed in latent space
%since the latent process maps from R^F \to R^E, we have MxFxE latent
%variables. Over dispersed control variables? Also, might be better to
%itialize as subset of initial qx, mean or uniformly? 
%TO DO: Implement a Gaussian eigenfunction version in matlab? At least in
%1D the X seem normally distributed.
z = randn(M, F, E); z(:, E+1:F, :) = 10/sqrt(12) * z(:, E+1:F, :);

p = struct('hyp', hyp, 'qx', qx, 'z', z);

clf
%define optimizer q
opt = struct('method', opt_method, 'length', -50, 'verbosity', 3);
[a b c] = minimize(rmfield(p,'qx'), 'wrap', opt, train, p.qx);
a.qx = p.qx;
[a.hyp.l]

clf
opt = struct('length',-30,'method',opt_method,'verbosity',3);
for n = 1:N
  vgpt(a, train(1:N))
  a.qx(n) = minimize(a.qx(n), 'vgpt', opt, train(n), rmfield(a, 'qx'));   
end

clf
opt = struct('length',-1000,'method',opt_method,'verbosity',3);
[a b c] = minimize(a, 'vgpt', opt, train); 

%[sd, so] = convert(qx.s(:,:,5));