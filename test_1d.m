function test_1d(n_training, n_seed, n_predict, data_std, M)
%Here we have two timeseries, one with a full amount of data and one serving
%as a test seed timeseries.

%--------------------------------------------------------------------------------
%Parameters that can be tweaked:

NUM_TRAINING_POINTS     = n_training; %Number of points that we have data for
NUM_POINTS_IN_TEST_SEED = n_seed;   
NUM_POINTS_TO_PREDICT   = n_predict;
DATA_STD_DEV            = data_std;  %Standard deviation of noise in data generative process
VAR_0                   = 1;   %Initial choice for variance of q_x distribution (isotropic)

if nargin < 5
M                       = (NUM_TRAINING_POINTS / 2);
end

%------------------------------------------------------------------------
%Generate training and testing samples:
x_0 = 0;
T_y = NUM_TRAINING_POINTS;
g_s = DATA_STD_DEV;
w = x_0;
x = zeros(T_y, 1);
x(1, 1) = x_0;
obs= zeros(T_y, 1);
obs(1, 1) = normrnd(x_0, g_s);

for t = 2:T_y
    w = f_smooth(w);
    x(t, 1) = w;
    obs(t, 1) = normrnd(w, g_s);
end

T_p = NUM_POINTS_IN_TEST_SEED;
T_p2 = NUM_POINTS_TO_PREDICT + T_p;
x_0 = 0;
w = x_0;
test_x = zeros(T_p2, 1);
test_x(1, 1) = x_0;
test_obs= zeros(T_p2, 1);
test_obs(1, 1) = normrnd(x_0, g_s);
for t = 2:T_p2
    w = f_smooth(w);
    test_x(t, 1) = w;
    test_obs(t, 1) = normrnd(w, g_s);
end
%---------------------------------------------------------------------------
%Package the input up into struct
%Hyp-----
hyp = struct('l', {[normrnd(0, 1e-2, 1, 1); 10]}, 'pn', {0}); %don't consider
                                                              %u at the moment
%s-------
s = zeros(2, 1, T_y); %initialise cov matrix
var = VAR_0; %variance of the gaussians
s(1, 1, :) =  (var/2)^(0.5) + normrnd(0, 1e-6); %break symmetry
tmp = [ones(1, 1, T_y / 2); -1 * ones(1, 1, T_y / 2)]; %alternating signs make
s(2, 1, :) = tmp(:) * (var/2)^(0.5) + normrnd(0, 1e-6);  %cov matrix noisy
                                                           %isotropic 
%m------
m = [obs']; %set means as observed outputs
%Other things for the initial training--
qx_1 = struct('m', {m}, 's', {s});
z = zeros(M, 2, 1);
z(:, 1, 1) = linspace(min(x) - 1, max(x) + 1, M);
y = obs(1:T_y);
u = zeros(T_y, 1); %Keep control inputs constant for now
data_1 = struct('y', {y}, 'u', {u});
p_1 = struct('hyp', hyp, 'qx', qx_1, 'z', z);
%%----------------------------------------------------------------------------
%Fit for the initial training
train_out = minimize(p_1, @vgpt, -3000, data_1);
[nlml dnlm lat] = vgpt(train_out, data_1);
%--------------------------------------------------------------------------
%Set parameters for the prediction inference on the seed timeseries
data_2 = struct('y', {test_obs(1:T_p)}, 'u', {u(1:T_p, :)});
qx_2 = struct('m', {test_obs(1:T_p)'}, 's', {s(:, :, 1:T_p)});
p_2 = struct('qx', qx_2);
test_in = struct('hyp', train_out.hyp, 'z', train_out.z);

%Optimise on the test seed observations
inf_out = minimize(p_2, @vgpt, -2000, data_2, test_in);
[nlml2 dnlm2 lat2] = vgpt(inf_out, data_2, test_in);

%Set parameters for the prediction, using the final state from the seed
predict_in = struct('hyp', train_out.hyp, 'z', train_out.z);
Sd = convert(inf_out.qx.s); Sd = Sd(:,:,end);
p_3 = struct('m', inf_out.qx.m(:, end), 'Sd', Sd);
data_3 = struct('u', 0);
%Analytically Predict-----------
first_3 = vgpt(p_3, data_3, predict_in);
%----------
[Sd, So] = convert(inf_out.qx.s);

bootstrap_series = struct('m',{[inf_out.qx.m first_3.m]}, 'Sd', ...
                          {cat(3, Sd, first_3.Sd)}, 'So', {cat(3, So, first_3.So)});
in = first_3;
for n = 1:T_p2-T_p - 1
    out = vgpt(in, data_3, predict_in);
    bootstrap_series = struct('m',{[bootstrap_series.m out.m]}, ...
    'Sd', {cat(3, bootstrap_series.Sd, out.Sd)}, 'So', {cat(3, bootstrap_series.So, out.So)});
    in = struct('m', out.m, 'Sd', out.Sd);
end

%---------------------------------------------------------------------------
%Plotting the training timeseries
plot_grid = [-3:0.1:2];

%Extract convariances
[Sd So] = convert(train_out.qx(1).s);

%Map back from latent space to observation space
adj_mean = conv_lat(lat.C, train_out.qx(1).m);
m_std = std(train_out.qx(1).m);
adj_z = conv_lat(lat.C, train_out.z(:, 1)')';

drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1), 'linewidth', ...
                           3, 'color', 'k');

%Plot the GP in latent space, map back to observation space
%Get a random subset of the inducing inputs, so we don't have more
%inputs than actual points.

meanfunc = [];
covfunc = @covSEiso;
sparse_cov = {@apxSparse, {covfunc}, train_out.z(:, 1)};
inf = @(varargin) infGaussLik(varargin{:}, struct('s', 0.0));
likfunc = @likGauss;
gp_hyp = struct('mean', [], 'cov', [train_out.hyp.l(1), log(m_std)], 'lik', train_out.hyp.pn);
[mu s2] = gp(gp_hyp, inf, meanfunc, sparse_cov, likfunc, train_out.qx(1).m(:, ...
                                                  1:T_y-1)', train_out.qx(1).m(:, ...
                                                  2:T_y)', train_out.z(:, ...
                                                  1));
mu = conv_lat(lat.C, mu')';
adj_s2 = lat.C(1) * sqrt(s2);

subplot(2, 2, 1);
f = [mu+2*adj_s2; flipdim(mu-2*adj_s2,1)];
fill([adj_z; flipdim(adj_z,1)], f, [7 7 7]/8)
hold on; plot(adj_z, mu, 'r', 'LineWidth', 2); 

%Plot the Gaussians

for t = 2:1:T_y
    mean = [adj_mean(1, t-1); adj_mean(1,t)];
    sigma = [Sd(t-1) So(t-1); So(t-1) Sd(t)];
    plot_2d_norm(mean, sigma, lat.C(1));
    plot(obs(t-1), obs(t), 'k+');
    drawArrow([obs(t-1) mean(1)], [obs(t) mean(2)])
end

plot(plot_grid, f_smooth(plot_grid), 'k', 'LineWidth', 2);
grid on;
% 
%Plotting the real-space trajectories of the output 
%with a predicted output

subplot(2, 2, 3);

hold on;
start = T_y - 20;

%Uncertainty in where we are is used
to_plot_Sd = reshape(Sd(1, 1, start-1:T_y-1), 1, T_y - start + 1);
to_plot_Sd_1 = sqrt(to_plot_Sd * lat.C(1).^2 + lat.R.^2);

f = [conv_lat(lat.C, train_out.qx(1).m(start:T_y)) - 2 * to_plot_Sd_1, ...
     fliplr(conv_lat(lat.C, train_out.qx(1).m(start:T_y)) + 2 * to_plot_Sd_1)];
hold on; 
fill([start:T_y, fliplr(start:T_y)], f, [7 7 7] / 8);
plot(start:T_y, adj_mean(start:T_y), 'r-+', 'LineWidth', 2);
plot(start:T_y, obs(start:T_y), 'k-+', 'LineWidth', 2);
plot(start:T_y, x(start:T_y), 'b.', 'LineWidth', 2);
% plot(T_1:T_n, conv_lat(lat.C, train_out.qx(1).m(T_1:T_n) + to_plot_Sd), 'r', ...
%      'LineWidth', 2);
% plot(T_1:T_n, conv_lat(lat.C, train_out.qx(1).m(T_1:T_n) - to_plot_Sd), 'r', ...
%       'Linewidth', 2);
y1=get(gca,'ylim');
plot([T_y T_y],y1, 'r:', 'LineWidth', 2);
legend('Predicted 2stdev', 'Mean Predicted', 'Observation', ['Actual ' ...
                    'state'], 'Observations stopped', 'location', ...
       'northwest')
title(sprintf(['Optimisation with %d time points, ' ...
               'obtaining nlml of %0.5g'], T_y, nlml))

%--------------------------------------------------------------------------------
%Plotting the generated test sequence:

plot_grid = [-3:0.1:2];
pred_out.m = bootstrap_series.m;
pred_lat = lat;

%Extract convariances
Sd = bootstrap_series.Sd;
So = bootstrap_series.So;

%Map back from latent space to observation space
adj_mean = conv_lat(pred_lat.C, pred_out.m);

drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1), 'linewidth', ...
                           3, 'color', 'k');

%Plot the GP in latent space, map back to observation space

subplot(2, 2, 2);
for t = 2:1:T_p2
    mean = [adj_mean(1, t-1); adj_mean(1,t)];
    sigma = [Sd(t-1) So(t-1); So(t-1) Sd(t)];
    plot_2d_norm(mean, sigma, pred_lat.C(1));
    text(mean(1), mean(2), num2str(t))
    plot(test_obs(t-1), test_obs(t), 'k+');
    drawArrow([test_obs(t-1) mean(1)], [test_obs(t) mean(2)]);
end

plot(plot_grid, f_smooth(plot_grid), 'k', 'LineWidth', 2);
grid on;
% 
%Plotting the real-space trajectories of the output 
%with a predicted output

subplot(2, 2, 4);

hold on;
%Uncertainty in where we are is used
to_plot_Sd = reshape(Sd(1, 1, 1:T_p2), 1, T_p2);
to_plot_Sd_2 = sqrt(to_plot_Sd * pred_lat.C(1).^2 + lat.R.^2);
hold on; 
f = [adj_mean(1:T_p2) - 2 * to_plot_Sd_2, ...
     fliplr(adj_mean(1:T_p2) + 2 * to_plot_Sd_2)];
hold on; 
fill([1:T_p2, fliplr(1:T_p2)], f, [7 7 7] / 8);
plot(1:T_p2, adj_mean(1:T_p2), 'r-+', 'LineWidth', 2);
plot(1:T_p2, test_obs(1:T_p2), 'k-+', 'LineWidth', 2);
plot(1:T_p2, test_x(1:T_p2), 'b.', 'LineWidth', 2);
% plot(T_1:T_n, conv_lat(pred_lat.C, train_out.qx(2).m(T_1:T_n) + to_plot_Sd), 'r', ...
%      'LineWidth', 2);
% plot(T_1:T_n, conv_lat(pred_lat.C, train_out.qx(2).m(T_1:T_n) - to_plot_Sd), 'r', ...
%       'Linewidth', 2);
%plot(T_1:T_n, adj_mean(T_1:T_n), 'b--', 'LineWidth', 2);

y1=get(gca,'ylim');
plot([T_p T_p],y1, 'r:', 'LineWidth', 2);
legend('Predicted 2stdev', 'Mean Predicted', 'Observation', ['Actual ' ...
                    'state'], 'Observations stopped', 'location', ...
       'southwest')
title(sprintf(['Predicting %d timesteps forward from a seed timeseries of ' ...
               '%d points'], T_p2 -T_p, T_p))

function out = conv_lat(C, points);
%Converts points in a latent space back to observation space
%  C       DxE+1   matrix mapping from latent space to observed space
%  point   ExK   points in the latent space
%  out     DxK   corresponding points in the observation space


out = C * [points; ones(1, size(points, 2))];

function plot_2d_norm(mean, cov, scale)
%Given a 2d vec mean and a 2x2 matrix cov, plot the normal
%distribution described

%Code taken from http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

[eigenvec eigenval] = eig(cov);

% Get the index of the largest eigenvector
[largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));
largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);

% Get the largest eigenvalue
largest_eigenval = max(max(eigenval));

% Get the smallest eigenvector and eigenvalue
if(largest_eigenvec_ind_c == 1)
    smallest_eigenval = max(eigenval(:,2));
    smallest_eigenvec = eigenvec(:,2);
else
    smallest_eigenval = max(eigenval(:,1));
    smallest_eigenvec = eigenvec(1,:);
end

% Calculate the angle between the x-axis and the largest
% eigenvector
angle = atan2(largest_eigenvec(2), largest_eigenvec(1));

% This angle is between -pi and pi.
% Let's shift it such that the angle is between 0 and 2pi
if(angle < 0)
    angle = angle + 2*pi;
end

% Get the 95% confidence interval error ellipse
chisquare_val = 2.4478 * scale;
a=chisquare_val*sqrt(largest_eigenval);
b=chisquare_val*sqrt(smallest_eigenval);
theta_grid = linspace(0,2*pi);
phi = angle;
X0=mean(1);
Y0=mean(2);

% the ellipse in x and y coordinates
ellipse_x_r  = a*cos( theta_grid );
ellipse_y_r  = b*sin( theta_grid );

%Define a rotation matrix
R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];

%let's rotate the ellipse to some angle phi
r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;

% Draw the error ellipse
plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'r-')
plot(mean(1), mean(2), 'rx')
hold on;

function [y] = f_smooth(x);
y = 0.8 + (x + 0.2) .* (1 - 5 ./ (1 + exp(-2 * x)));

    
function [Sd, So] = convert(s)
[t, E, T]= size(s); Sd = zeros(E,E,T); So = zeros(E,E,T-1);
for t = 1:T, Sd(:,:,t) = s(:,:,t)'*s(:,:,t); end               % diagonal terms
for t = 2:T, So(:,:,t-1) = s(:,:,t-1)'*s(:,:,t); end             % off-diagonal