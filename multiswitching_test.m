%MULTISWITCHING_TEST test script for multiswitching control
% This m-file solves the parabolic multiswitching control problem
%  min 1/2 \|y-yd\|^2 + \alpha/2 \int_0^T |u(t)|_1^2 \,dt
%      s.t. y_t - \Delta y = Bu, \partial_\nu y = 0, y(0) = 0
% using the approach described in the paper
%  "A convex analysis approach to multiswitching control of partial
%  differential equations"
% by Christian Clason, Armin Rund, Karl Kunisch and Richard C. Barnard,
% http://math.uni-graz.at/mobis/publications/SFB-Report-2015-005.pdf.
%
% March 23, 2015                        Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>
%                          Richard C. Barnard <richard.c.barnard@gmail.com>

% problem statement
d.alpha = 1e-2;                 % quadratic penalty
d.Nf = 7;                       % number of control functions
obs = 'x.^2 + y.^2 <= 0.5^2';   % observation domain

% parameters of the optimization method:
d.maxit_ssn  = 30;      % maximal iterations for SSN (semismooth Newton method)
d.abstol_ssn = 1e-9;    % absolute tolerance of SSN
d.reltol_ssn = 1e-6;    % relative tolerance of SSN
d.maxit_cg   = 50;      % maximal iterations for CG
d.reltol_cg  = 1e-6;    % relative tolerance of CG

% discretization:
d.h = 0.1;                      % mesh size parameter hmax
d.Nt = 201;                     % number of time points
d.Nc = d.Nt-1;                  % number of degrees of freedom for control
d.Tend = 10;                    % terminal time
d.t = linspace(0,d.Tend,d.Nt);  % time discretization
d.tau = d.t(2)-d.t(1);          % time step size (equidistant grid)

% assemble finite element matrices:
[p,e,t] = initmesh('squareg','Hmax',d.h);
[d.A,d.M,~]  = assema(p,t,1,1,0);   % stiffness matrix A and mass matrix M
[~,d.Mobs,~] = assema(p,t,0,obs,0); % mass matrix Mobs of observation domain
d.Nx = size(d.M,1);                 % number of discrete points in space
d.MpA = d.M + 0.5*d.tau*d.A;        % system matrices of CG(1)DG(0) Crank-Nicolson
d.MmA = d.M - 0.5*d.tau*d.A;

% assemble control operator and desired state vector
d.Bu = [];
d.yd = zeros(d.Nx,d.Nt);
for n = 1:d.Nf
    phi = pi/4 + (n-1)*2*pi/d.Nf;
    x = cos(phi)/sqrt(2);   y = sin(phi)/sqrt(2);
    str_fn = strcat('(x-(',num2str(x),')).^2 + (y-(',num2str(y),')).^2');
    [~,~,fn] = assema(p,t,0,0,strcat(str_fn,'<= 0.1^2'));
    d.Bu = [d.Bu,fn];
    
    xi = cos(n+d.t);
    [~,~,g] = assema(p,t,0,0,str_fn);
    g = d.M\g;
    h = xi.*sin(2*pi*d.t /d.Tend).^2;
    d.yd = d.yd + g*h;
end

%% compute control

q = zeros(d.Nf*d.Nc,1);
for k = 2:12  % homotopy in gamma
    d.gamma = 10^(-k);
    
    % semi-smooth Newton method working on q
    [q,output] = ssn(q,d);
    
    % check convergence
    if output.flag > 1
        fprintf('\n#### reverting to last gamma=%1.0e\n', gamma_old);
        q = q_old;  d.gamma = gamma_old;
        break;
    else
        q_old = q;  gamma_old = d.gamma;
    end
    
    % compute the control u
    [j,G,u,D,d_vec] = objfun(q,d);
    fprintf('   |I|_{1,...,Nf} = %s\n\n',sprintf('%3d  ',histc(d_vec,1:d.Nf)));
    
    % plot control
    t = linspace(0,d.Tend,d.Nc);
    up = reshape(u,d.Nc,d.Nf);  up(up==0) = NaN; % don't plot zeros
    dp = d_vec;
    figure(1);  plot(t,up,'LineWidth',1.5);
    hold on;
    dp(d_vec ~= 2) = NaN;  dp(d_vec == 2) = 0;  % 2 components active
    plot(t,dp,'o','Color','k','MarkerSize',2);  % 3+ components active
    dp(d_vec == 2) = NaN;  dp(d_vec > 2) = 0;
    plot(t,dp,'x','Color','k','MarkerSize',2);
    hold off;
    title(['optimal control for \gamma = ', num2str(d.gamma,'%1.0e')]);
    figure(2);  stem(t,d_vec);
    title(['number of active components for \gamma = ', num2str(d.gamma,'%1.0e')]);
    drawnow update;
end
