 
 
clear;
seed = 2022;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
mu = 1e-3;
L = eigs(A'*A, 1);
x0 = randn(n, 1);
 
opts = struct();
opts.method = 'grad_huber';
opts.verbose = 0;
opts.maxit = 4000;
opts.ftol = 1e-5;
opts.alpha0 = 1 / L;
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);
opts.verbose = 0;
opts.maxit = 500;
if opts.verbose
    fprintf('mu=1e-3\n');
end
[x, out] = LASSO_con(x0, A, b, mu, opts);
 t=length(x);
figure
 
xa=1:t;
%subplot(2,2,1)
semilogy(xa,x , '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
 
title("u=1e-3加BB")
ylabel(' x^k  ', 'fontsize', 14, 'interpreter', 'latex');
xlabel('n');

[x1, out1] = LASSO_con(x0, A, b, mu, opts);
f_star1 = min(out1.fvec);
opts.verbose = 0;
opts.maxit = 400;
if opts.verbose
    fprintf('mu=1e-3\n');
end
[x1, out1] = LASSO_con(x0, A, b, mu, opts);
data11 = (out1.fvec - f_star)/f_star;
k11 = min(length(data11),400);
data11 = data11(1:k11)
hold on
%%%
% 将 $\mu$ 修改为 |1e-2| 重复实验。
% mu = 1e-2;
% opts = struct();
% opts.method = 'grad_huber';
% opts.verbose = 0;
% opts.maxit = 4000;
% opts.ftol = 1e-8;
% opts.alpha0 = 1 / L;
% [x, out] = LASSO_con(x0, A, b, mu, opts);
% f_star = min(out.fvec);
% 
% opts.verbose = 0;
% opts.maxit = 400;
% if opts.verbose
%     fprintf('\nmu=1e-2\n');
% end
% [x, out] = LASSO_con(x0, A, b, mu, opts);
% data2 = (out.fvec - f_star)/f_star;
% k2 = min(length(data2),400);
% data2 = data2(1:k2);
 fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on

semilogy(0:k11-1, data11, '-','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('加BB', '无BB');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步 u=1e-2');
print(fig, '-depsc','grad.eps');

% figure;
% subplot(2, 1, 1);
% plot(u, 'Color',[0.2 0.1 0.99], 'Marker', 'x', 'LineStyle', 'none');
% xlim([1, 1024]);
% title('精确解');

% subplot(2, 1, 2);
% plot(x, 'Color',[0.2 0.1 0.99], 'Marker', 'x', 'LineStyle', 'none');
% xlim([1, 1024]);
% title('梯度法解');

saveas(gcf, 'solu-smoothgrad.eps');
 