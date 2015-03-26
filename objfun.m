function [obj,grad,u,DH,d_vec,as_hash] = objfun(q,d)
% OBJFUN compute functional value, gradient
% [OBJ,GRAD,U,DH,D_VEC,AS_HASH] = OBJFUN(Q,D) computes the value OBJ of the
% functional to be minimized together with the gradient GRAD in the dual
% point Q. U is the control obtained from the regularized subdifferential,
% whose Newton derivative at Q is DH. D_VEC and AS_HASH contain information
% about the active set for monitoring convergence. The structure D contains
% the problem parameters.
%
% March 23, 2015                        Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>
%                          Richard C. Barnard <richard.c.barnard@gmail.com>

%% compute regularized subdifferential H_gamma(q), Newton derivative DH_gamma(q)

% compute proximal mapping w := prox(q), Newton derivative Dw = D prox(q)
v  = reshape(q,d.Nc,d.Nf)';   % rows are components
w  = v;                       % identity on some parts
Dw = speye(size(q,1));
as = zeros(d.Nc,d.Nf);        % active set, only for output
d_vec = zeros(1,d.Nc);        % number of active components, only for output
for i = 1:d.Nc                % loop over time
    [sorted,ix] = sort(abs(v(:,i)),'descend'); % sort components by magnitude
    found = false;
    for dd = 1:d.Nf-1         % find number of active control components
        if sorted(dd+1) < d.alpha/(dd*d.alpha+d.gamma)*sum(sorted(1:dd))
            found = true;
            break;
        end
    end
    if ~found                  % all components are active
        dd = d.Nf;
    end
    ind = ix(1:dd);
    w(ind,i) = sign(v(ind,i))*d.alpha/(dd*d.alpha+d.gamma)*sum(sorted(1:dd));
    for j = ind
        Dw((j-1)*d.Nc+i, (ind-1)*d.Nc+i) = ...
            (d.alpha/(dd*d.alpha+d.gamma))*sign(v(j,i)*v(ind,i)'); %#ok<SPRIX>
    end
    d_vec(i) = dd;            % dd controls are active at t(i)
    as(i,ind) = 1;            % controls u_{ix(1:dd)} are active at t(i)
end
w = reshape(w',d.Nc*d.Nf,1);

% compute subdifferential H(q) = (q-prox(q))/gamma, Newton derivative
Hq = (q-w)/d.gamma;
DH = (speye(size(q,1)) - Dw)/d.gamma;

% active sets: hash for active components
as_hash = zeros(d.Nc,1);
for n = 1:d.Nf
    as_hash = as_hash + 2^(n-1)*as(:,n);
end

%% compute residual, objective

% control u = h_gamma(q)
u = Hq;

% solve state equation
y = zeros(d.Nx,d.Nt);         % zero initial condition
for m = 1:d.Nt-1              % time stepping
    y(:,m+1) = d.MpA \ (d.MmA*y(:,m) + d.tau*d.Bu*u(m:d.Nc:end));
end
res = y - d.yd;               % residual

% summed trapezoidal rule for the tracking term (scaling later)
normres  = diag(res'*d.Mobs*res);
tracking = sum(normres) - 0.5*(normres(1)+normres(end));
% L2(l1)-norm for control costs (scaling later)
ccosts   = sum(sum(reshape(abs(u),d.Nc,d.Nf),2).^2);
% L2(l2)-norm for Moreau-Yosida regularization (scaling later)
mycosts  = sum(u.^2);

obj = 0.5*d.tau*(tracking + d.alpha*ccosts + d.gamma*mycosts);

%% compute gradient

BTz = 0*q;                                   % initialization of B'*z

% solve adjoint equation for adjoint state z
z = d.MpA \ (0.5*d.tau*d.Mobs*res(:,d.Nt));  % terminal condition
BTz(d.Nc:d.Nc:end) = d.Bu'*z;
for m = d.Nc-1:-1:1                          % time stepping (backward)
    z = d.MpA \ (d.MmA*z + d.tau*d.Mobs*res(:,m+1));
    BTz(m:d.Nc:end) = d.Bu'*z;
end

grad = q + BTz;
