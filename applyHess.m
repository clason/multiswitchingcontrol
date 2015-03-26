function Hdq = applyHess(dq,DH,d)
% APPLYHESS compute application of Hessian
% HDQ = APPLYHESS(DQ,DH,D) computes the action HDQ of the Hessian in 
% direction DQ. DH contains the Newton derivative of the regularized
% subdifferential at Q. The structure D contains the problem parameters.
%
% March 23, 2015                        Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>
%                          Richard C. Barnard <richard.c.barnard@gmail.com>

% linearized control
du = DH*dq;

% solve linearized state equation
dy = zeros(d.Nx,d.Nt);        % zero initial condition
for m = 1:d.Nt-1              % time stepping
    dy(:,m+1) = d.MpA \ (d.MmA*dy(:,m) + d.tau*d.Bu*du(m:d.Nc:end));
end

% solve linearized adjoint equation for dz
BTdz = 0*dq;                   % initialization of B'*dz
dz = d.MpA \ (0.5*d.tau*d.Mobs*dy(:,d.Nt));  % terminal condition
BTdz(d.Nc:d.Nc:end) = d.Bu'*dz;
for m = d.Nc-1:-1:1           % time stepping (backward)
    dz = d.MpA \ (d.MmA*dz + d.tau*d.Mobs*dy(:,m+1));
    BTdz(m:d.Nc:end) = d.Bu'*dz;
end

% Hessian direction
Hdq = dq + BTdz;
