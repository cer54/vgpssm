%A minimal usage, with a pre-trained model a and data test, train.
%Infers hidden states from the unseen test sequence test(1) and 
%then predicts the following observations given the control inputs.

qx.m = a.qx(1).m(:,1:5);      % initialise with something random of
qx.s = a.qx(1).s(:,:,1:5);
tt.y = test(1).y(1:5,:);          % get the first 5 observations
tt.u = test(1).u(1:4,:);         % and the first 4 actions
vgpt(a, train); %initialise Psi, C,R, into memory
opt = struct('length', -2000, 'method', 'BFGS', 'verbosity', 3);
qx = minimize(qx, 'vgpt', opt, tt, rmfield(a,'qx'));       % infer hidden states
E = 5;
u_in = struct('u', test(1).u(5:end,:));
outs = vgpt(qx, u_in, a);
for n=1:size(outs.m, 2)
    real_errors(:, n) = 2*sqrt(diag(outs.Sd(:,:,n)));
end

%Plot
titles = {'Position', 'Sin Angle', 'Cos Angle', 'Velocity', 'Angular Velocity'}
lims = [[-1 1]; [-1.2 1.2]; [-1.2 1.2]; [-3 3]; [-5 15]];
legend('Inferred or Predicted Coordinate', 'Observed Coordinate')
for e=1:E
   subplot(E, 1, e);
   hold on;
   title(titles{e});
   errorbar(6:size(test(1).y, 1), outs.m(e, :),  real_errors(e, :), 'k-');
   plot(test(1).y(:, e), 'r--')
   ylim(lims(e, :))
end