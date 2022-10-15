 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 可以直接利用带BB步长的梯度下降法最小化 $f$来得到原问题解的一个近似解（解的质量与 $\sigma$ 选取有关）。
% 
% 为了加快算法的收敛速度，我们采用连续化策略来调整正则化参数 $\mu$，详见 <..\LASSO_con\LASSO_con.html LASSO问题连续化策略>。
%
%% 初始化和迭代准备
% 输入信息： 模型参数 $A$, $b$, $\mu$，迭代初始值 $x^0$，原问题对应的正则化系数 $\mu_0$，包含算法参数的结构体 |opts|。
%
% 输出信息： 迭代得到的解（原 LASSO 问题，即正则化参数为 $\mu_0$，且不使用光滑化）和包含迭代信息的结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的原始 LASSO 问题目标函数值（对应于原问题的 $\mu_0$）
% * |out.fval| ：迭代终止时的原始 LASSO 问题目标函数值（对应于原问题的 $\mu_0$）
% * |out.tt| ：运行时间
% * |out.itr| ：表示迭代次数
% * |out.flag| ：标记是否达到收敛
function [x, out] = LASSO_grad_huber_inn(x, A, b, mu, mu0, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的停机准则，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对梯度的停机准则，当当前步梯度范数小于该值时认为该条件满足
% * |opts.alpha0| ：步长的初始值
% * |opts.sigma| ：Huber 光滑化参数 $\sigma$
% * |opts.verbose| ：不为 0 时输出每步迭代信息，否则不输出
if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
tt = tic;

%%%
% 梯度的计算：对于函数 $h(x):=\frac{1}{2}\|Ax-b\|^2_2$，其梯度 $g=A^\top (Ax-b)$。
%
% 另一方面， |huber_g| 通过将 $\|x\|_1$ 小于阈值的部分以二次函数近似，求近似后的梯度来实现梯度法。
% |idx| 表示分量小于阈值的下标，这些分量对应的梯度为 $x_i/\sigma$，
% 其余为 $\mathrm{sign}(x_i)$。两部分梯度求和，得到光滑化函数 $f$的梯度。
r = A * x - b;
g = A' * r;

huber_g = sign(x);
idx = abs(x) < opts.sigma;
huber_g(idx) = x(idx) / opts.sigma;

g = g + mu * huber_g;
nrmG = norm(g,2);
%%%
% 计算 $x^0$ 处光滑化函数值，三项分别对应于 $h(x^0)$, $L_{\sigma}(x^0)$
% 中小于阈值的部分（二次函数部分）， $L_{\sigma}(x^0)$ 中大于等于阈值的部分（线性部分）。
f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2*opts.sigma)) + sum(abs(x(abs(x) >= opts.sigma)) - opts.sigma/2));

%%% 
% 初始设定。
out = struct();
%%%
% 注意 |out.fvec| 中记录的是原始问题目标函数的值。
out.fvec = .5*norm(r,2)^2 + mu0*norm(x,1);

%%%
% 从 |opts.alpha0| 中取得起始步长， $\eta$为线搜索条件未满足时步长的衰减率。
alpha = opts.alpha0;
eta = 0.2;

%%%
% 线搜索参数。
rhols = 1e-6;
gamma = 0.85;
Q = 1;
Cval = f;

%% 迭代主循环
% 对于连续化策略下的每个正则化系数 $\mu$，使用梯度下降法求解 $f$的最小值，以 |opts.maxit| 为最大迭代次数。
for k = 1:opts.maxit
    %%%
    % 记录上一次迭代的信息。
    fp = f;
    gp = g;
    xp = x;
    %%%
    % 线搜索循环选取合适步长并更新迭代点 $x$，并且在新的 $x$ 处更新各变量值（更新方式与初始化时相同，参考上文）。
    %
    % 重置线搜索次数为 1。
    nls = 1;
    while 1
        
        x = xp - alpha*gp;
        r = A * x - b;
        g = A' * r;
        huber_g = sign(x);
        idx = abs(x) < opts.sigma;
        huber_g(idx) = x(idx) / opts.sigma;
        f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2*opts.sigma)) + sum(abs(x(abs(x) >= opts.sigma)) - opts.sigma/2));
        g = g + mu * huber_g;
        
        %%%
        % 线搜索准则 (Zhang & Hager) $f(x^k+\alpha d^k)\le C_k+\rho\alpha (g^k)^\top d^k$
        % 或进行超过 10 次步长衰减后退出线搜索。
        % 在当前步长不符合线搜索条件的情况下，对当前步长以 $\eta$ 进行衰减，线搜索次数加一。
        if f <= Cval - alpha*rhols*nrmG^2 || nls >= 10
            break
        end
        alpha = eta*alpha;
        nls = nls+1;
    end
    
    %%%
    % 线搜索结束，得到更新的 $x$, $g$。计算梯度范数和原始目标函数值。 |fvec| 
    % 记录每一步对应的原 LASSO 问题的目标函数值（即正则化系数为 $\mu_0$，且不适用光滑化）。并进行内层循环的收敛判断：
    % 若当前梯度小于阈值或者目标函数变化小于阈值，内层迭代终止。
    
    nrmG = norm(g,2);
    forg = .5*norm(r,2)^2 + mu0*norm(x,1);
    out.fvec = [out.fvec, forg];
    
    %%%
    % 详细输出模式下打印每一次迭代信息。
    if opts.verbose
        fprintf('%4d\t %.4e \t %.1e \t %.2e \t %2d \n',k, f, nrmG, alpha, nls);
    end

    if nrmG < opts.gtol || abs(fp - f) < opts.ftol
        break;
    end
    
    %%%
    % 计算 BB 步长作为下一步迭代的初始步长。令 $s^k=x^{k+1}-x^k$, $y^k=g^{k+1}-g^k$，
    % 这里在偶数与奇数步分别对应 $\displaystyle\frac{(s^k)^\top s^k}{(s^k)^\top y^k}$
    % 和 $\displaystyle\frac{(s^k)^\top y^k}{(y^k)^\top y^k}$ 两个 BB 步长。
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
    %%%
    % 计算 (Zhang & Hager) 线搜索准则中的递推常数，其满足 $C_0=f(x^0),\ C_{k+1}=(\gamma
    % Q_kC_k+f(x^{k+1}))/Q_{k+1}$ ，序列 $Q_k$ 满足 $Q_0=1,\ Q_{k+1}=\gamma
    % Q_{k}+1$。
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
end
%%%
% 向外层迭代（连续化策略）报告内层迭代的退出方式，当达到最大迭代次数退出时， |out.flag|
% 记为 1，否则则为达到收敛标准，记为 0。 这个指标用于判断是否进行正则化系数的衰减。并记录输出。
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
%% 参考页面
% 该函数由连续化策略调用，关于连续化策略参见 <..\LASSO_con\LASSO_con.html LASSO问题连续化策略>
% 。另外，我们将在 <.\demo.html 实例：梯度法解LASSO问题>
% 中构造一个 LASSO 问题，并展示此算法的应用和效果。
%
% 此页面的源代码请见： 
% <../download_code/lasso_grad/LASSO_grad_huber_inn.m
% LASSO_grad_huber_inn.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将
