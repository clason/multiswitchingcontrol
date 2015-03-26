function [x, flag, relres, iter] = pcg_ip(A, b, rtol, maxit, D)
%PCG_IP conjugate gradient method with active set inner product
% [X, FLAG, RELRES, ITER] = PCG_IP(A, B, RTOL, MAXIT, D) solves AX=B using
% conjugate gradient method with inner product <x,y> = x'*D*y. The
% iteration is terminated if the relative residual is smaller than RTOL or 
% if MAXIT iterations are reached. The achieved residual Euclidean norm is 
% returned as RELRES and the number of iterations as ITER. FLAG is 0 if the
% iteration converged and 1 if the maximum number of iterations is reached.
%
% March 23, 2015                        Armin Rund <armin.rund@uni-graz.at>
%                            Christian Clason <christian.clason@uni-due.de>
%                          Richard C. Barnard <richard.c.barnard@gmail.com>

% active set inner product
ip = @(x,y) x'*(D*y);      

% initialize
x = 0*b;  r = b;  d = r;  delta = ip(r,r);
res0 = sqrt(delta);  iter = 0;  flag = 0;

% CG iteration
while  sqrt(delta) > res0*rtol
    if iter == maxit
        flag = 1;
        break;
    end
    iter = iter + 1;
    
    Ad = A(d);
    
    gamma = ip(Ad,d);
    alpha = delta / gamma;
    
    x = x + alpha*d;
    r = r - alpha*Ad;
    
    deltaold = delta;
    delta = ip(r,r);
    
    beta = -alpha*ip(Ad,r)/deltaold;
    d    = r + beta*d;
end

% relative residual in standard norm (used for checking convergence)
relres = norm(r)/res0;
