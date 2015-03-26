function [q,output] = ssn(q,d)
%SSN semismooth Newton method
% [Q,OUTPUT] = SSN(Q,D) computes the optimal dual variable Q from a given
% initial point using a semismooth Newton method. The structure D contains
% the problem parameters, while OUTPUT is a structure containing the
% following data:
%     j:     optimal value,
%     g0:    residual norm in optimality conditions
%     ssnit: number of Newton steps
%     cgit:  number of conjugate gradient steps in last Newton iteration
%     flag:  0 - converged with relative tolerance, 1 - converged with
%            absolute tolerance, 2 - diverged (too many iterations)
%
% March 23, 2015                        Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>
%                          Richard C. Barnard <richard.c.barnard@gmail.com>

fprintf('Starting SSN for gamma=%1.0e:      (CG flag 0: converged, 1: max iterations)\n',d.gamma);
fprintf('Iter    objective   |I| _{1,2,3}  | normgrad   dAS | stepsize flag relres CGit\n');

%% semismooth Newton iteration

ssnit = 0;  GGold = 1e99;  as_old = zeros(d.Nc,1);  tau = 1;
while ssnit <= d.maxit_ssn
    % compute new gradient
    [j,G,~,D,d_vec,as] = objfun(q,d);
    if ssnit == 0
        G0 = sqrt(d.tau)*norm(G);  output.g0 = G0;  output.j0 = j;
        flag = 0;  cgit = 0;
    end
    
    % line search on gradient norm (correctly scaled discrete norm)
    GG = sqrt(d.tau)*norm(G);
    if GG >= GGold       % if no decrease: backtrack (never on iteration 1)
        tau = tau/2;
        q = q - tau*dq;
        if tau < 1e-7    % if step too small: terminate Newton iteration
            fprintf('\n#### not converged: step size too small\n');
            output.flag = 3;
            break;
        else             % else: bypass rest of loop; backtrack further
            continue;
        end
    end
    
    % compute statistics and change in active sets
    I_vec = histc(d_vec,1:d.Nf);      % number of points in each active set
    as_change = nnz(abs(as - as_old)>0.5);  % number of points that changed
    
    % output iteration details
    fprintf('%3d:  %1.5e  %3d  %3d  %3d  | %1.3e  ', ...
        ssnit, j,  I_vec(1),  I_vec(2),  I_vec(3), GG);
    if ssnit > 0
        fprintf('%3d | %1.1e   %d  %1.1e    %d\n', as_change, tau, flag, relres, cgit);
    else
        fprintf('\n');
    end
    
    % terminate Newton?
    if (GG < d.reltol_ssn*sqrt(G0)) && (as_change == 0)  % convergence (relative norm)
        fprintf('\n#### converged with relative tol: |grad|<=%1.1e |grad0|\n',d.reltol_ssn);
        output.flag = 0;
        break;
    elseif (GG < d.abstol_ssn) && (as_change == 0)  % convergence (absolute norm)
        fprintf('\n#### converged with absolute tol: |grad|<=%1.1e\n',d.abstol_ssn);
        output.flag = 1;
        break;
    elseif ssnit == d.maxit_ssn                     % failure, too many iterations
        fprintf('\n#### not converged: too many iterations\n');
        output.flag = 2;
        break;
    end
    % otherwise update information, continue
    ssnit = ssnit+1;  GGold = GG;  as_old = as;  tau = 1;
    
    % compute Newton step, update
    DG = @(dq) applyHess(dq,D,d);  % Hessian
    [dq, flag, relres, cgit] = pcg_ip(DG, -G, d.reltol_cg, d.maxit_cg, D);
    q = q + dq;
end

%% output

output.j     = j;
output.g     = GG;
output.ssnit = ssnit;
output.cgit  = cgit;
