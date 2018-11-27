T=100;
E=1;
U=0;
drift = @kink_transition;
drift_factor = 1;
latent_noise =0.1;
observation_noise = .1;
N = 1;
u = zeros(T,U);
x = zeros(N,T,E);
for n = 1:N
    x_0 = randn(1,E);
    x(n,1,:)=x_0;
    for t = 2:T+1
        x(n,t,:) =  x(n,t-1,:)+ drift_factor* drift(x(n,t-1,:))+latent_noise*randn(1,E);
    end
end
figure
clf
plot(x)
figure
scatter(x(:,1:T),x(:,2:T+1))%-x(:,1:T))
y = x+observation_noise*randn(size(x));
train_latent = transpose(x(:,2:T+1));
train_y = transpose(y(:,2:T+1));
train_u = u;
train = struct('y',train_y,'u',train_u,'latent',train_latent);
save("1d_test_data.mat",'train')








clf
plot(x); hold on; plot(y); hold off;
function [dif_x] = kink_transition(cur_x)
    if cur_x<2
        dif_x = .2;
    else
        dif_x= -cur_x;
    end
end