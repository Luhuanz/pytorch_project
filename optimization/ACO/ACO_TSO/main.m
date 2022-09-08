clear;
clc;
%x=[51 27 56 21 4 6 58 71 54 40 94 18 89 33 12 25 24 58 71 94 17 38 13 82 12 58 45 11 47 4 ]';
%y=[14 81 67 92 64 19 98 18 62 69 30 54 10 46 34 18 42 69 61 78 16 40 10 7 32 17 21 26 35 90 ]';
%position = [x, y]; 真实数据
position = 100 * randn(40, 2);  %虚拟数据城市位置

epochs = 50;
ants = 50;
alpha = 1.4; % 表示信息素重要程度系数
beta = 2.2;  % 距离重要程度系数
rho = 0.15; % 信息素蒸发因子 类似香水散去的权重  
Q = 10^6;  % 信息素增加强度系数
cities = size(position, 1); %表示问题的规模 ，这里指的是城市个数
% 城市之间的距离矩阵
Distance = ones(cities, cities); %完全图
for i = 1: cities
    for j = 1: cities
        if i ~= j
            Distance(i, j) = ((position(i, 1) - position(j, 1))^2 + (position(i, 2) - position(j, 2))^2)^0.5;  
        else
            Distance(i, j) = eps;  %i=j时不计算，应该为0，但后面的距离要取倒数，用eps（浮点相对精度）表示
        end
        Distance(j, i) = Distance(i, j);%对称矩阵
    end
end
Eta = 1./Distance;    %Eta为启发因子，这里设为距离的倒数 1/d_it(t)
Tau = ones(cities, cities);  %Tau为信息素矩阵 40x40
% 每只蚂蚁的路线图
Route = zeros(ants, cities);  %存储并记录路径的生成
epoch = 1; % 迭代
% 记录每回合最优城市
R_best = zeros(epochs, cities); %每次迭代蚂蚁群中最优路线
L_best = inf .* ones(epochs, 1); % 每次迭代蚂蚁群中最佳长度
L_ave = zeros(epochs, 1); % 每次迭代平均长度
% 开始迭代
while epoch <= epochs
    % 随机位置
    %ants 50  随机放到 40个城市上
    RandPos = []; 
    for i = 1: ceil(ants / cities)
        RandPos = [RandPos, randperm(cities)];
    end
    Route(:, 1) = (RandPos(1, 1:ants))';
    % 这个50只蚂蚁按概率p选择下一座城市
    for j = 2:cities
        for i = 1: ants
            Visited = Route(i, 1:j-1); %记录访问过的城市，防止重复
            NoVisited = zeros(1, (cities - j + 1)); % 待访问的城市
            P = NoVisited;  %带访问的城市选择概率分布
            num = 1;
            for k = 1: cities
                if length(find(Visited == k)) == 0
                    NoVisited(num) = k;
                    num = num + 1; %每访问一个城市数+1
                end
            end
          %计算待选城市的概率分布
            for k = 1: length(NoVisited) 
                P(k) = (Tau(Visited(end), NoVisited(k))^alpha) * (Eta(Visited(end), NoVisited(k))^beta);
            end
            P = P / sum(P);
               %按概率原则选取下一个城市
            Pcum = cumsum(P);  %cumsum，元素累加即求和
            select = find(Pcum >= rand);%若计算的概率大于原来的就选择这条路线
            to_visit = NoVisited(select(1));
            Route(i, j) = to_visit;
        end
    end
       
    if epoch >= 2
        Route(1, :) = R_best(epoch - 1, :);
    end
    %%4.记录本次迭代最佳路线
    Distance_epoch = zeros(ants, 1);
    for i = 1: ants
        R = Route(i, :);
        for j = 1: cities - 1
            Distance_epoch(i) = Distance_epoch(i) + Distance(R(j), R(j + 1)); %开始距离为0，m*1的列向量
        end
        Distance_epoch(i) = Distance_epoch(i) + Distance(R(1), R(cities)); %一轮下来后走过的距离
    end 
    L_best(epoch) = min(Distance_epoch); %最佳距离取最小
    pos = find(Distance_epoch == L_best(epoch));
    R_best(epoch, :) = Route(pos(1), :);%此轮迭代后的最佳路线
    L_ave(epoch) = mean(Distance_epoch); %此轮迭代后的平均距离
    epoch = epoch + 1;
    
    Delta_Tau = zeros(cities, cities); %开始时信息素为n*n的0矩阵
    for i = 1: ants
        for j = 1: (cities - 1)
            Delta_Tau(Route(i, j), Route(i, j + 1)) = Delta_Tau(Route(i, j), Route(i, j + 1)) + Q / Distance_epoch(i);
                 %此次循环在路径（i，j）上的信息素增量
        end
        Delta_Tau(Route(i, 1), Route(i, cities)) = Delta_Tau(Route(i, 1), Route(i, cities)) + Q / Distance_epoch(i);
          %此次循环在整个路径上的信息素增量
    end
    Tau = (1 - rho) .* Tau + Delta_Tau; %考虑信息素挥发，更新后的信息素
    Route = zeros(ants, cities);%直到最大迭代次数
end
%% 结果展示
Pos = find(L_best == min(L_best));
Short_Route = R_best(Pos(1), :);
Short_Length = L_best(Pos(1), :);
figure
% subplot(121);
DrawRoute(position, Short_Route);
% subplot(122);
% plot(L_best);
% hold on
% plot(L_ave, 'r');
% title('平均距离和最短距离');
