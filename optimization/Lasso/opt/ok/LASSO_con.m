 
function [x, out] = LASSO_con(x0, A, b, mu0, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 3000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-5; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-3; end
if ~isfield(opts, 'factor'); opts.factor = 0.1; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'mu1'); opts.mu1 = 100; end
if ~isfield(opts, 'gtol_init_ratio'); opts.gtol_init_ratio = 1/opts.gtol; end
if ~isfield(opts, 'ftol_init_ratio'); opts.ftol_init_ratio = 1e5; end
if ~isfield(opts, 'opts1'); opts.opts1 = struct(); end
if ~isfield(opts, 'etaf'); opts.etaf = 1e-1; end
if ~isfield(opts, 'etag'); opts.etag = 1e-1; end
L = eigs(A'*A,1);
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1/L; end

if ~isfield(opts, 'method'); error('Need opts.method'); end
algf = eval(sprintf('@LASSO_%s_inn',opts.method));
 
out = struct();
out.fvec = [];
k = 0;
x = x0;
mu_t = opts.mu1; % 大的正则项warmup
tt = tic;
 
f = Func(A, b, mu_t, x);
opts1 = opts.opts1;
opts1.ftol = opts.ftol*opts.ftol_init_ratio;
opts1.gtol = opts.gtol*opts.gtol_init_ratio;
out.itr_inn = 0;
 
while k < opts.maxit
    
    opts1.maxit = opts.maxit_inn;
    opts1.gtol = max(opts1.gtol * opts.etag, opts.gtol);
    opts1.ftol = max(opts1.ftol * opts.etaf, opts.ftol);
    opts1.verbose = opts.verbose > 1;
    opts1.alpha0 = opts.alpha0;
    if strcmp(opts.method, 'grad_huber'); opts1.sigma = 1e-3*mu_t; end
    fp = f;
    [x, out1] = algf(x, A, b, mu_t, mu0, opts1);
    f = out1.fvec(end);
    out.fvec = [out.fvec, out1.fvec];
    k = k + 1;
    nrmG = norm(x - prox(x - A'*(A*x - b),mu0),2);
    if opts.verbose
        fprintf('itr: %d\tmu_t: %e\titr_inn: %d\tfval: %e\tnrmG: %.1e\n', k, mu_t, out1.itr, f, nrmG);
    end
    if ~out1.flag
        mu_t = max(mu_t * opts.factor, mu0);
    end

    if mu_t == mu0 && (nrmG < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
  
    out.itr_inn = out.itr_inn + out1.itr;
end
out.fval = f;
out.tt = toc(tt);
out.itr = k;
 
    function f = Func(A, b, mu0, x)
        w = A * x - b;
        f = 0.5 * (w' * w) + mu0 * norm(x, 1);
    end
    function y = prox(x, mu)
        y = max(abs(x) - mu, 0);
        y = sign(x) .* y;
    end
end
 