%% 初始化种群  
clear
clc
%% Sphere
 
f= @(x) x .* sin(x) .* cos(2 * x) - 2 * x .* sin(3 * x) +3 * x .* sin(4 * x); % 函数表达式    % 求这个函数的最大值  


N = 20;                         % 初始种群个数  
d = 1;                          % 可行解维数  
ger = 100;                      % 最大迭代次数       
limit = [0, 50];               % 设置位置参数限制  
vlimit = [-10, 10];               % 设置速度限制  
w = 0.8;                        % 惯性权重  
c1 = 0.5;                       % 自我学习因子  
c2 = 0.5;                       % 群体学习因子   
figure(1);
ezplot(f,[0,0.01,limit(2)]);   %曲线

x = limit(1) + (  limit( 2 ) -  limit( 1)  ) .* rand(N, d);%初始种群的位置  

v = rand(N, d);                  % 初始种群的速度  
xm = x;                          % 每个个体的历史最佳位置  
ym = zeros(1, d);                % 种群的历史最佳位置  
fxm = ones(N, 1)*inf;               % 每个个体的历史最佳适应度  
fym = inf;                       % 种群历史最佳适应度  
hold on  
plot(xm, f(xm), 'ro');title('初始状态图');  
 pause(2);
figure(2);  
%% 群体更新  
iter = 1;  
record = zeros(ger, 1);          % 记录器  
while iter <= ger  
     fx = f(x) ; % 个体当前适应度     
     for i = 1:N         % 每只鸟
        if fx(i)  <fxm(i) 
            fxm(i) = fx(i);     % 更新个体历史最佳适应度  
            xm(i,:) = x(i,:);   % 更新个体历史最佳位置(取值)   鸟的x位置坐标 
        end   
     end  
    if  min(fxm)  < fym 
        [fym, nmin] = min(fxm);   % 更新群体历史最佳适应度  
        ym = xm(nmin, :);      % 更新群体历史最佳位置  
    end  
    v = v * w + c1 * rand * (xm - x) + c2 * rand * (repmat(ym, N, 1) - x);% 速度更新  
    % 边界速度处理  
    v(v > vlimit(2)) = vlimit(2);  
    v(v < vlimit(1)) = vlimit(1);  
    x = x + v;% 位置更新  
    % 边界位置处理  
    x(x > limit(2)) = limit(2);  
    x(x < limit(1)) = limit(1);  
    record(iter) = fym;%最大值记录  
    x0 = 0 : 0.01 : limit(2);  
    subplot(1,2,1)
    plot(x0, f(x0), 'b-', x, f(x), 'ro');title('状态位置变化')
    subplot(1,2,2);plot(record);title('最优适应度进化过程')  
    pause(0.01)  
    iter = iter+1;  

end  

%x0 = 0 : 0.01 : limit(2);  
%figure(3);plot(x0, f(x0), 'b-', x, f(x), 'ro');title('最终状态位置')  
disp(['最大值：',num2str(fym)]);  
disp(['变量取值：',num2str(ym)]);  