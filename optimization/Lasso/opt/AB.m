function [x, f_o,dg,iter,f_] = AB(x, A, b, mu, mu0,maxit,fto1,gtol,alpha0,sigma)
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
%f_ = f;
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

end
dg=dg2;
iter=k;
f_=f;

end     