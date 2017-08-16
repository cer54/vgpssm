function [nlml] = compute_nlml(p, data)
% Variational GP Timeseries inference. Compute the nlml lower bound and its
% derivative wrt hyp hyperparameters, qx distribution and z inducing inputs.  
%Given two arguments, we expect the following
% p                  parameter struct
%   hyp     1 x E    GP hyperparameter struct
%     l     F x 1    log length scale
%     pn    1 x 1    log process noise std dev
%   qx      1 x N    struct array for Gaussian q(x) distribution
%     m     E x T_n  mean
%     s  2ExE x T_n  representation of covariance
%   z     M x F x E  inducing inputs
%   C       D x E    emission matrix (optional)
%   R       D x D    emission noise covariance (optional)
% data      1 x N    data struct
%   y     T_n x D    cell array of observations
%   u     T_n x U    cell array of control inputs
%

% Copyright (C) 2017 by Carl Edward Rasmussen

N = length(p.qx); z = p.z; [M, F, E] = size(z); D = size(data(1).y,2); hyp = p.hyp;
u = arrayfun(@(x)(x.u),data,'UniformOutput',false); [qx(1:N).m] = deal(p.qx(:).m);
y = arrayfun(@(x)(x.y),data,'UniformOutput',false);
for n = 1:N, [qx(n).Sd, qx(n).So] = convert(p.qx(n).s); end % convert covariance

T = sum(arrayfun(@(x)size(x.m,2),qx));
L1 = Psi(hyp, qx, z, u, 0); 
L2 = 0; L3 = 0; 
for n = 1:N
    L2 = L2 + sum(qx(n).m(:,2:end).^2,2) + diag(sum(qx(n).Sd(:,:,2:end),3)); 
    L3 = L3 + gaussMarkovEntropy(qx(n).Sd, qx(n).So);
end
L4 = -(T-N)*sum([hyp(:).pn])-(T-N)*E*log(2*pi)/2;
L5 = -exp(-2*[hyp(:).pn]) * (L2+T-N) / 2;
L6 = likelihood(y, qx, p);
nlml = -L1-L3-L4-L5-L6;

  
function [lml, dnlml] = Psi(hyp, qx, z, u, test);
% hyp       1 x E    GP hyperparameter struct
%   l       F x 1    log length scale
%   pn      1 x 1    log process noise std dev
% qx        1 x N    Gaussian q(x) distribution
%   m     E x T_n    mean
%   Sd  ExE x T_n    diagonal elements of covariance matrix
%   So  ExE x T_n-1  immediately off-diagonal elements of covariance matrix
% z       M x F x E  inducing inputs
% u       T_n x U    cell array of control inputs 
% lml       1 x 1    contribution to the log marginal likelihood
% dnlml              derivatives
% test               flag indicating test mode

persistent K Psi1 Psi2;                        % keep these around if necessary
[M, F, E] = size(z);                                                % get sizes

if nargin < 5 %Use to return stored Psi values
  lml = Psi1; dnlml = Psi2;
else
  if ~test %If we test, we use the stored values of Psi1, Psi2 from the training set.
    K = zeros(M,M,E); Psi1 = zeros(M,E); Psi2 = zeros(M,M,E);
  end
  lml = 0;
  Sd = zeros(F,F);
  for e = 1:E                                                       % for each GP
    K(:,:,e) = exp(-maha(z(:,:,e),[],diag(exp(-2*hyp(e).l)))/2) + 1e-6*eye(M);
    iL = diag(exp(-hyp(e).l)); L2 = diag(exp(2*hyp(e).l));
    b1 = zeros(M,1); b2 = zeros(M,M);
    for n = 1:length(qx)                                   % for each time series
      for t = 2:size(qx(n).m, 2)                             % for each time step
        Sd(1:E,1:E) = qx(n).Sd(:,:,t-1);          % covariance in top left corner
        r1 = prod(diag(chol(eye(F)+iL*Sd*iL)));                        % sqrt det
        r2 = prod(diag(chol(eye(F)+2*iL*Sd*iL)));                      % sqrt det
        s = bsxfun(@minus, z(:,:,e), [qx(n).m(:,t-1)' u{n}(t-1,:)]);
        a = s/(L2+Sd);
        b1 = b1 + (qx(n).m(e,t) + a(:,1:E)*qx(n).So(:,e,t-1)) ...
                                                        .*exp(-sum(a.*s,2)/2)/r1;
        b2 = b2 + exp(-maha(s,-s,inv(L2+2*Sd))/4) / r2;
      end
    end
    if test
      w = (K(:,:,e)+Psi2(:,:,e))\Psi1(:,e);
      W = -K(:,:,e)\Psi2(:,:,e)/(Psi2(:,:,e)+K(:,:,e)) + w*w';
      lml = lml + exp(-2*hyp(e).pn)*(b1'*w ...
                         - sum(sum(b2.*exp(-maha(z(:,:,e),[],inv(L2))/4).*W))/2);
    else
      Psi1(:,e) = b1 * exp(-2*hyp(e).pn);
      Psi2(:,:,e) = b2 * exp(-2*hyp(e).pn) .* exp(-maha(z(:,:,e),[],inv(L2))/4);
      lml = lml - sum(log(diag(chol(K(:,:,e)+Psi2(:,:,e))))) + ...
             sum(log(diag(chol(K(:,:,e))))) + trace(K(:,:,e)\Psi2(:,:,e))/2 + ...
                                   Psi1(:,e)'/(K(:,:,e)+Psi2(:,:,e))*Psi1(:,e)/2;
    end
  end

  if nargout > 0
    dnlml.z = zeros(M,F,E);
    for n = 1:size(qx,2), dnlml.qx(n).m = 0*qx(n).m; dnlml.qx(n).So = 0*qx(n).So;
    dnlml.qx(n).Sd = -0*qx(n).Sd; end;
    for e = 1:E
      w = (K(:,:,e)+Psi2(:,:,e))\Psi1(:,e);
      R = K(:,:,e)\Psi2(:,:,e);
      W = -R/(Psi2(:,:,e)+K(:,:,e)) + w*w';
      dnlml.hyp(e).pn = 2*sum(w.*Psi1(:,e)) - sum(sum(W.*Psi2(:,:,e)));
      iL = diag(exp(-hyp(e).l)); L2 = diag(exp(2*hyp(e).l));
      W1 = W .* exp(-maha(z(:,:,e),[],inv(L2))/4);
      D = zeros(M,F); H = zeros(F,1);
      for n = 1:length(qx)
        T = size(qx(n).m,2);
        A = zeros(E,T); B = zeros(E,E,T); C = zeros(E,E,T-1);
        for t = 2:T
          Sd(1:E,1:E) = qx(n).Sd(:,:,t-1);         % covariance in top left corner
          r2 = prod(diag(chol(eye(F)+2*iL*Sd*iL)));                     % sqrt det
          s = bsxfun(@minus, z(:,:,e), [qx(n).m(:,t-1)' u{n}(t-1,:)]);
          a = s/(L2+Sd);
          a2 = s/(2*L2+4*Sd);
          SiS = (L2(1:E,1:E)+Sd(1:E,1:E))\qx(n).So(:,e,t-1);
          r = exp(-sum(a.*s,2)/2) / prod(diag(chol(eye(F)+iL*Sd*iL)));
          g = (qx(n).m(e,t) + a(:,1:E)*qx(n).So(:,e,t-1)).*w.*r;
          W2 = W1.*exp(-maha(s,-s,inv(L2+2*Sd))/4);
          X = bsxfun(@plus,permute(a2,[1 3 2]),permute(a2,[3 1 2]));
          A(:,t-1) = A(:,t-1) + SiS*(w'*r) - a(:,1:E)'*g + ...
                            squeeze(sum(sum(bsxfun(@times,W2,X(:,:,1:E)),2),1))/r2;
          A(e,t) = -w'*r;
          B(:,:,t-1) = squeeze(sum(sum(bsxfun(@times, ...
             bsxfun(@times,W2,X(:,:,1:E)),permute(X(:,:,1:E),[1 2 4 3])),2),1))/r2;
          B(:,:,t-1) = B(:,:,t-1) + SiS*((w.*r)'*a(:,1:E)) ...
                                        + inv(L2(1:E,1:E)+Sd(1:E,1:E))*sum(g)/2 ...
                                        - a(:,1:E)'*bsxfun(@times,g,a(:,1:E))/2 ...
                                - inv(L2(1:E,1:E)+2*Sd(1:E,1:E))*sum(sum(W2))/r2/2;
          C(:,e,t-1) = -bsxfun(@times,a(:,1:E),r)'*w;
          if ~test
            D(:,1:E) = D(:,1:E) - bsxfun(@times,w,r)*SiS';
            D = D + bsxfun(@times,g,a) - W2*a2/r2 - bsxfun(@times,sum(W2,2),a2)/r2;
            H = H + diag(Sd/(L2+2*Sd))*sum(sum(W2))/r2 ...
               + exp(2*hyp(e).l).*squeeze(sum(sum(bsxfun(@times,W2,X.^2),2),1))/r2;
            H(1:E) = H(1:E) ...
                  + 2*exp(2*hyp(e).l(1:E)).*diag(SiS*bsxfun(@times,w,r)'*a(:,1:E));
            H = H - diag(Sd/(L2+Sd))*sum(g);
            H = H - exp(2*hyp(e).l).*diag(a'*bsxfun(@times,g,a));
          end
        end
        dnlml.qx(n).m = dnlml.qx(n).m + A * exp(-2*hyp(e).pn); 
        dnlml.qx(n).Sd = dnlml.qx(n).Sd + B * exp(-2*hyp(e).pn);
        dnlml.qx(n).So = dnlml.qx(n).So + C * exp(-2*hyp(e).pn);  
      end
      if ~test
      G = W.*Psi2(:,:,e);
      a = z(:,:,e)*diag(exp(-2*hyp(e).l)/2);
      dnlml.z(:,:,e) = D*exp(-2*hyp(e).pn) + G*a - bsxfun(@times,sum(G,2),a);
      B = bsxfun(@minus,permute(a,[1 3 2]),permute(a,[3 1 2]));
      dnlml.hyp(e).l = H * exp(-2*hyp(e).pn)  ...
               + exp(2*hyp(e).l).*squeeze(sum(sum(bsxfun(@times,G,B.^2),1),2));
      G = (R/(K(:,:,e)+Psi2(:,:,e))*R' + w*w').*K(:,:,e);
      a = z(:,:,e)*diag(exp(-2*hyp(e).l));
      B = bsxfun(@minus,permute(z(:,:,e),[1 3 2]),permute(z(:,:,e),[3 1 2]));
      dnlml.hyp(e).l = dnlml.hyp(e).l ...
            + exp(-2*hyp(e).l).*squeeze(sum(sum(bsxfun(@times,B.^2,G),1),2))/2;
      dnlml.z(:,:,e) = dnlml.z(:,:,e) + G*a - bsxfun(@times,sum(G,2),a);
      end
    end
  end
end
function [L dLd dLo] = gaussMarkovEntropy(d, o);
[E, E, T] = size(d); dd = zeros(T,1); dp = zeros(T-1,1); 
for t = 1:T, dd(t) = det(d(:,:,t)); end                      % det of diagonals
for t = 1:T-1, dp(t) = dd(t)*det(d(:,:,t+1)-o(:,:,t)'/d(:,:,t)*o(:,:,t)); end;
L = E*T*(1+log(2*pi))/2 + sum(log(dp))/2 - sum(log(dd(2:T-1)))/2;     % entropy
if nargout > 1                                              % want derivatives?
  dLd = zeros(E,E,T); dLo = zeros(E,E,T-1);
  for t = 1:T-1
    dLd(:,:,t) = dLd(:,:,t) + inv(d(:,:,t)-o(:,:,t)/d(:,:,t+1)*o(:,:,t)')/2;
    dLd(:,:,t+1) = inv(d(:,:,t+1)-o(:,:,t)'/d(:,:,t)*o(:,:,t))/2;
    dLo(:,:,t) = -d(:,:,t)\o(:,:,t)/(d(:,:,t+1)-o(:,:,t)'/d(:,:,t)*o(:,:,t));
  end
  for t = 2:T-1, dLd(:,:,t) = dLd(:,:,t) - inv(d(:,:,t))/2; end
end

function [L, R_out, C_out, dm, dS] = likelihood(y, qx, p)
persistent C R
if nargin < 3
    L = C; R_out = R; %If we just want to fetch the stored C,R.
else
    D = size(y{1},2); E = size(qx(1).m,1); T = sum(arrayfun(@(x)size(x.m,2),qx)); 
    N = size(y,2); yy = zeros(D); ym = zeros(D, E+1); mm = zeros(E+1); d = zeros(1,N);
    for n = 1:N
        m = [qx(n).m' ones(size(qx(n).m,2),1)]; mm = mm + m'*m; ym = ym + y{n}'*m;
        yy = yy + y{n}'*y{n}; mm(1:E,1:E) = mm(1:E,1:E) + sum(qx(n).Sd,3); 
    end
    C_star = ym/mm; R_star = (yy - C_star*ym')/T;
    if ~isfield(p, 'C') && ~isfield(p, 'R') % then use optimal C, R
        C = C_star; R = R_star;
        L = -D*T*(1+log(2*pi))/2 - T*sum(log(diag(chol(R))));          % log likelihood
    else
        if ~isfield(p, 'C')
            C = C_star; R = p.R;
        elseif ~isfield(p, 'R')
            C = p.C; R = R_star;
        else
            C = p.C; R = p.R;
        end
        for n=1:N
            w = y{n}' - C*[qx(n).m; ones(1,size(qx(n).m,2))];
            d(1,n) = sum(sum(inv(R).*(w*w' + C(:,1:E)*sum(qx(n).Sd,3)*C(:,1:E)')));
        end
        L = -D*T*log(2*pi)/2 - T*sum(log(diag(chol(R)))) - sum(d)/2;            
    end  
    C_out = C; R_out = R;
    if nargout > 3                                        % do we want derivatives?
        dm = cell(N,1);
        for n = 1:N,
            dm{n} = C(:,1:E)'/R*(y{n}'-C*[qx(n).m; ones(1,size(qx(n).m,2))]);
        end
        dS = -C(:,1:E)'/R*C(:,1:E)/2;            % all dS identical, return once only
    end
end
function [Sd, So] = convert(s)
[t, E, T]= size(s); Sd = zeros(E,E,T); So = zeros(E,E,T-1);
for t = 1:T, Sd(:,:,t) = s(:,:,t)'*s(:,:,t); end               % diagonal terms
for t = 2:T, So(:,:,t-1) = s(:,:,t-1)'*s(:,:,t); end             % off-diagonal

function r = revert(s, dSd, dSo)
for t = 1:size(s,3), r(:,:,t) = s(:,:,t)*(dSd(:,:,t)+dSd(:,:,t)'); end
for t = 2:size(s,3)
  r(:,:,t-1) = r(:,:,t-1) + s(:,:,t)*dSo(:,:,t-1)';
  r(:,:,t) = r(:,:,t) + s(:,:,t-1)*dSo(:,:,t-1);
end

% Squared Mahalanobis distance (a-b)*Q*(a-b)'; vectors are row-vectors
% a, b  d x n  matrices containing n length d row vectors
% Q     d x d  weight matrix
% K     n x n  squared distances
function K = maha(a, b, Q)                         
if isempty(b), b = a; end
aQ = a*Q; K = bsxfun(@plus,sum(aQ.*a,2),sum(b*Q.*b,2)')-2*aQ*b';

function qx_opt = predict_forwards(hyp, qx, z, u)
%Given a timeseries corresponding to latent states of an observed
%timeseries, analytically calculate the optimal next step params
%Fetch the Psi1,2 values by calling Psi with 4 arguments
[Psi1, Psi2] = Psi(hyp, qx, z, u);
E = size(qx, 2); [M, F, E] = size(z); Q = zeros(E, E);
b3 = zeros(E,1);
for e = 1:E
  K(:,:,e) = exp(-maha(z(:,:,e),[],diag(exp(-2*hyp(e).l)))/2) + ...
      1e-6*eye(M);
  w = (K(:,:,e)+Psi2(:,:,e))\Psi1(:,e);
  iL = diag(exp(-hyp(e).l)); L2 = diag(exp(2*hyp(e).l));
  Sd = zeros(F,F);
  Sd(1:E,1:E) = qx.Sd(:,:,1);          % covariance in top left corner
  r1 = prod(diag(chol(eye(F)+iL*Sd*iL)));                        % sqrt det
  s = bsxfun(@minus, z(:,:,e), [qx.m(:,1)' u(1,:)]); 
  a = s/(L2+Sd);
  b3(e,1) = w'*(exp(-sum(a.*s,2)/2)/r1); 
  b4 = (bsxfun(@times, a(:,1:E), exp(-sum(a.*s,2)/2)/r1))'*w;
  b5(:, e) = b4;
  Q(e, e) = exp(2 * hyp(e).pn);
end

qx_opt.m = b3;
qx_opt.So = qx.Sd*b5;
qx_opt.Sd = Q + b5'*qx.Sd(:, :, end)*b5;

