 
clear, clc;
rng(2022); % 设置随机数种子
%%%% 初始化 %%%%
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;

mu = 1e-3;
sigma=1e-3*mu;
L = eigs(A'*A, 1);%最大特征值
x = randn(n, 1);%x0
alpha0 = 1 / L;
fgood=mu*norm(u,1);
 maxit = 20000;
 gtol = 1e-6;
 ftol = 1e-8 ;
 verbose = 0 ;
tt = tic;
 mu0=mu;
r = A * x - b;
g = A' * r;

huber_g = sign(x);
idx = abs(x) <  sigma;
huber_g(idx) = x(idx) / sigma;

g = g + mu * huber_g;
nrmG = norm(g,2);
 
f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2* sigma)) + sum(abs(x(abs(x) >=  sigma)) -  sigma/2));

 
fvec = .5*norm(r,2)^2 + mu0*norm(x,1);

 
alpha =  alpha0;
 
 
beta=0.25; % 线条件未满足时步长的衰减率；类似warmup 策略
c=0.2;
 
for k = 1:maxit
 
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
        idx = abs(x) < sigma;
        huber_g(idx) = x(idx) / sigma;
        f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2* sigma)) + sum(abs(x(abs(x) >=  sigma)) -  sigma/2));
        g = g + mu * huber_g;
        
        if  f <= f1 - alpha*c*nrmG ^2 ||nls>=30  % d=-1方向
            break
        end
        alpha = beta*alpha;
        nls = nls+1;
    end
    
    
    
    nrmG = norm(g,2);
    forg = .5*norm(r,2)^2 + mu0*norm(x,1);
    fvec = [ fvec, forg];
    if nrmG <  gtol || abs(fp - f) <  ftol
        break;
    end
   
    % 计算 BB 步长作为下一步迭代的初始步长。 
    dx = x - xp;
    dg = g - gp;
    dxg = abs(dx'*dg);
    if dxg > 0
        if mod(k,2)==0
            alpha = dx'*dx/dxg;
        else
            alpha = dxg/(dg'*dg);
        end
        %%%
        % 保证步长的合理范围。
        alpha = max(min(alpha, 1e12), 1e-12);
    end
 
end
 
 
 
 