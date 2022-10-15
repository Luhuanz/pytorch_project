
function [x, out] = LASSO_grad_huber_inn(x, A, b, mu, mu0,opts)
 
if ~isfield(opts, 'maxit'); opts.maxit = 3000; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
tt = tic;
r = A * x - b;
g = A' * r;

huber_g = sign(x);
idx = abs(x) < opts.sigma;
huber_g(idx) = x(idx) / opts.sigma;

g = g + mu * huber_g;
nrmG = norm(g,2);
 
f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2*opts.sigma)) + sum(abs(x(abs(x) >= opts.sigma)) - opts.sigma/2));
out = struct();
out.fvec = .5*norm(r,2)^2 + mu0*norm(x,1);
alpha = opts.alpha0;
 
 beta=0.25; % 线条件未满足时步长的衰减率；类似warmup 策略
 c=0.2;
for k = 1:opts.maxit
    fp = f;
    gp = g;
    xp = x;
    nls = 1;
    while 1
        f1=f;
        x = xp - alpha*gp;
        r = A * x - b;
        g = A' * r;
        huber_g = sign(x);
        idx = abs(x) < opts.sigma;
        huber_g(idx) = x(idx) / opts.sigma;
        f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2*opts.sigma)) + sum(abs(x(abs(x) >= opts.sigma)) - opts.sigma/2));
        g = g + mu * huber_g;  
        if f <= f1 - alpha*c*nrmG ^2 ||nls>=30   
            break
        end
        alpha = beta*alpha;
        nls = nls+1;
    end    
    nrmG = norm(g,2);
    forg = .5*norm(r,2)^2 + mu0*norm(x,1);
    out.fvec = [out.fvec, forg];
    if opts.verbose
        fprintf('%4d\t %.4e \t %.1e \t %.2e \t %2d \n',k, f, nrmG, alpha, nls);
    end

    if nrmG < opts.gtol || abs(fp - f) < opts.ftol
        break;
    end
%     dx = x - xp;
%     dg = g - gp;
%     dxg = abs(dx'*dg);
%     if dxg > 0
%         if mod(k,2)==0
%             alpha = dx'*dx/dxg;
%         else
%             alpha = dxg/(dg'*dg);
%         end
%         alpha = max(min(alpha, 1e20), 1e-20);
%     end
  
end
 
if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = nrmG;
end
 