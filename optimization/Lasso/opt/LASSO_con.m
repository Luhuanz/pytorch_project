% 考虑 LASSO 问题
% * |out.fvec| ：每一步迭代的函数值 
% * |out.itr_inn| ：总内层迭代次数
% * |out.fval| ：迭代终止时的目标函数值
% * |out.tt| ：运行时间
% * |out.itr| ：外层迭代次数
function [x, out] = LASSO_con(x0, A, b, mu0, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大外层迭代次数
% * |opts.maxit_inn| ：最大内层迭代次数
% * |opts.ftol| ：针对函数值的停机判断条件
% * |opts.gtol| ：针对梯度的停机判断条件
% * |opts.factor| ：正则化系数的衰减率
% * |opts.verbose| ：不等于 0 时输出每步迭代信息，否则不输出
% * |opt.mu1| ：初始的正则化系数（采用连续化策略，从更大的正则化系数开始）
% * |opts.alpha0| ：初始步长
% * |opts.ftol_init_ratio| ：初始时停机准则 |opts.ftol| 的放大倍数
% * |opts.gtol_init_ratio| ：初始时停机准则 |opts.gtol| 的放大倍数
% * |opts.etaf| ：每步外层循环的停机判断标准 |opts.ftol| 的缩减
% * |opts.etag| ：每步外层循环的停机判断标准 |opts.gtol| 的缩减
% * |opts.opts1| ：结构体，用于向内层算法提供其它具体的参数
if ~isfield(opts, 'maxit'); opts.maxit = 30; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
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

%%%
% 通过 |opts.method| 选择求解正则化参数 $\mu_t$对应的子问题的算法。
if ~isfield(opts, 'method'); error('Need opts.method'); end
algf = eval(sprintf('@LASSO_%s_inn',opts.method));
%%%
% 添加所有子目录的路径到工作路径。
addpath(genpath(pwd));

%%%
% 迭代准备，注意到采取了连续化策略，因此 $\mu$ 以更大的 |opts.mu1| 开始。
out = struct();
out.fvec = [];
k = 0;
x = x0;
mu_t = opts.mu1;
tt = tic;
%%%
% 当前正则化系数 $\mu_t$ 和 $x$ 对应的函数值。
f = Func(A, b, mu_t, x);

%%%
% |opts1| 是固定某一正则化系数 $\mu_t$ 时，相应子问题求解算法的停机准则参数取值集合。 
% |opts1.ftol|
% 为停机判据中的函数相对变化的阈值。|opts1.gtol|
% 为梯度范数阈值。注意到它们都随着外层迭代的进行而逐渐减小，这要求子问题求解的精确度逐渐提高。
% 在初始时，这两个阈值均选择较大的值。
opts1 = opts.opts1;
opts1.ftol = opts.ftol*opts.ftol_init_ratio;
opts1.gtol = opts.gtol*opts.gtol_init_ratio;
out.itr_inn = 0;
%% 连续化循环
% 求解 LASSO 问题的经典技巧，通过采用连续化策略，从较大的 $\mu$ 逐渐减小到 $\mu_0$，
% 以加速收敛。对每个 $\mu_t$ 下调用 |opts.method| 来求解相应的子问题。
while k < opts.maxit
    %%%
    % 内层循环参数设置，记录在结构体 |opts1| 中
    %
    % * |opts1.itr|：最大迭代次数，由 |opts.maxit_inn| 给出
    % * |opts1.ftol|：针对函数值的迭代终止条件
    % * |opts1.gtol|：针对梯度的迭代终止条件
    % * |opts1.alpha0|：初始步长
    % * |opts1.verbose|：当 |ops.verbose| 大于 1 时为真，此时详细输出内层迭代的信息
    opts1.maxit = opts.maxit_inn;
    opts1.gtol = max(opts1.gtol * opts.etag, opts.gtol);
    opts1.ftol = max(opts1.ftol * opts.etaf, opts.ftol);
    opts1.verbose = opts.verbose > 1;
    opts1.alpha0 = opts.alpha0;
    
    %%%
    % 仅当 |opts.method| 为 |'grad_huber'| 时， |opts1.sigma| 给出 Huber 光滑化的范围；
    if strcmp(opts.method, 'grad_huber'); opts1.sigma = 1e-3*mu_t; end
    
    %%%
    % 调用内层循环函数，记录每一次内层循环的返回信息。
    % |out.fvec| 记录每一步的 $x$ 对应的原始函数值（正则化系数为 $\mu$ 而非当前 $\mu_t$）。
    fp = f;
    [x, out1] = algf(x, A, b, mu_t, mu0, opts1);
    f = out1.fvec(end);
    out.fvec = [out.fvec, out1.fvec];
    k = k + 1;
    
    %%%
    % 由于 $\ell_1$-范数不可导，这里 |nrmG| 表示 LASSO 问题的最优性条件的违反度来
    nrmG = norm(x - prox(x - A'*(A*x - b),mu0),2);
    
    %%%
    % 详细输出模式下打印每一次外层循环信息。
    if opts.verbose
        fprintf('itr: %d\tmu_t: %e\titr_inn: %d\tfval: %e\tnrmG: %.1e\n', k, mu_t, out1.itr, f, nrmG);
    end
    %%%
    % 当内层循环因达到收敛条件而退出时，缩减当前正则化系数 $\mu_t$，
    % 并判断收敛。外层循环的收敛条件：当 $\mu$ 已经减小到与 $\mu_0$
    % 相同并且函数值或梯度满足收敛条件时，停止外层循环。
    if ~out1.flag
        mu_t = max(mu_t * opts.factor, mu0);
    end

    if mu_t == mu0 && (nrmG < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
    %%%
    % 更新总迭代次数。
    out.itr_inn = out.itr_inn + out1.itr;
end

%%%
% 当外层循环终止时，记录当前的函数值、外层迭代次数和运行时间。
out.fval = f;
out.tt = toc(tt);
out.itr = k;
%% 辅助函数
%%%
% 原始 LASSO 问题的目标函数。
    function f = Func(A, b, mu0, x)
        w = A * x - b;
        f = 0.5 * (w' * w) + mu0 * norm(x, 1);
    end
%%%
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
    function y = prox(x, mu)
        y = max(abs(x) - mu, 0);
        y = sign(x) .* y;
    end
end
%% 参考页面
% 该连续化策略调用具体的某一算法的内部迭代器，具体包括：
%
% * <../lasso_grad/LASSO_grad_huber_inn.html LASSO
% 问题的梯度下降法>
% * <../lasso_subgrad/LASSO_subgrad_inn.html LASSO
% 问题的次梯度法>
% * <../lasso_proxg/LASSO_proximal_grad_inn.html LASSO
% 问题的近似点梯度法>
% * <../lasso_proxg/LASSO_Nesterov_inn.html LASSO 问题的
% FISTA 算法>
% * <../lasso_proxg/LASSO_Nesterov2nd_inn.html LASSO
% 问题的第二类 Nesterov 加速算法>
% * <../lasso_bcd/LASSO_bcd_inn.html LASSO
% 问题的分块坐标下降法>
% * <../lasso_dualalg/LASSO_pdhg_inn.html LASSO
% 问题的原始-对偶混合梯度算法>
%
% 此页面的源代码请见： <../download_code/LASSO_con/LASSO_con.m LASSO_con.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将