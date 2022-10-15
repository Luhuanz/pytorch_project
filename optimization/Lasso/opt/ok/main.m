
clear, clc;
rng(2022); % 设置随机数种子
%%%% 初始化 %%%%
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
mu0=1e-3;
aaa=20;
mu=100;
maxit = 3000;
global kk
kk=0;
global x
x = randn(n, 1);%x0
fgood=mu0*norm(u,1);% f*
global res 
for i =1:aaa % 让mu下降 mu=0.1*mu
    sigma=1e-3*mu;
    L = eigs(A'*A, 1);%最大特征值
    alpha0 = 1 / L;
    fto1=1e-5; % 函数值停止条件
    gtol=1e-3; % 梯度停止条件
    r=A*x-b;
    g=A'*r;
    h_g = sign(x);%dL
    idx = abs(x) <  sigma;
    h_g(idx) = x(idx) /sigma;
    g = g + mu * h_g; %导函数值
    dg2 = norm(g,2);% f导函数的2范数
    f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2*sigma)) + sum(abs(x(abs(x) >= sigma)) - sigma/2));%一维光滑后
    f_o=.5*norm(r,2)^2 + mu0*norm(x,1);% 记录原始函数值
    alpha=alpha0;
    beta=0.25; % 线条件未满足时步长的衰减率；类似warmup 策略
    c=0.2;
    nls=1;
    for k = 1:maxit
        f0 = f;
        g0 = g;
        x0 = x;
        nls=1;
        while 1
            f1=f;
            x = x0 - alpha*g0;
            r = A * x - b;
            g = A' * r;
            h_g = sign(x);
            idx = abs(x) < sigma;
            h_g(idx) = x(idx) /  sigma;
            f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2* sigma)) + sum(abs(x(abs(x) >=  sigma)) -  sigma/2));
            g = g + mu * h_g;
            if f <= f1 - alpha*c*dg2 ^2 ||nls>=30  % d=-1方向
                break
            end
            alpha = beta*alpha;
            nls = nls+1;
        end

        dg2= norm(g,2);
        forg = .5*norm(r,2)^2 + mu0*norm(x,1);
        f_o=[f_o,forg];
        %% stop条件
        if dg2<gtol||abs(f0-f)<fto1
            break;
        end
        %BB
        sx = x - x0; %函数值
        sg = g - g0;% 梯度
        dxg = abs(sx'*sg);
        if dxg > 0
            if mod(k,2)==0
                alpha = sx'*sx/dxg;
            else
                alpha = dxg/(sg'*sg);
            end
            alpha = max(min(alpha, 1e20), 1e-20);  % 条件
        end
     kk=kk+1;  
    end
    
    f=f_o(end);
    res=[res,f_o];
    i=i+1;
    mu=1e-1*mu;
    if mu<mu0
        break
    end  
end

 



        