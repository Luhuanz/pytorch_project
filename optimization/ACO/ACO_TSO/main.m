clear;
clc;
%x=[51 27 56 21 4 6 58 71 54 40 94 18 89 33 12 25 24 58 71 94 17 38 13 82 12 58 45 11 47 4 ]';
%y=[14 81 67 92 64 19 98 18 62 69 30 54 10 46 34 18 42 69 61 78 16 40 10 7 32 17 21 26 35 90 ]';
%position = [x, y]; ��ʵ����
position = 100 * randn(40, 2);  %�������ݳ���λ��

epochs = 50;
ants = 50;
alpha = 1.4; % ��ʾ��Ϣ����Ҫ�̶�ϵ��
beta = 2.2;  % ������Ҫ�̶�ϵ��
rho = 0.15; % ��Ϣ���������� ������ˮɢȥ��Ȩ��  
Q = 10^6;  % ��Ϣ������ǿ��ϵ��
cities = size(position, 1); %��ʾ����Ĺ�ģ ������ָ���ǳ��и���
% ����֮��ľ������
Distance = ones(cities, cities); %��ȫͼ
for i = 1: cities
    for j = 1: cities
        if i ~= j
            Distance(i, j) = ((position(i, 1) - position(j, 1))^2 + (position(i, 2) - position(j, 2))^2)^0.5;  
        else
            Distance(i, j) = eps;  %i=jʱ�����㣬Ӧ��Ϊ0��������ľ���Ҫȡ��������eps��������Ծ��ȣ���ʾ
        end
        Distance(j, i) = Distance(i, j);%�Գƾ���
    end
end
Eta = 1./Distance;    %EtaΪ�������ӣ�������Ϊ����ĵ��� 1/d_it(t)
Tau = ones(cities, cities);  %TauΪ��Ϣ�ؾ��� 40x40
% ÿֻ���ϵ�·��ͼ
Route = zeros(ants, cities);  %�洢����¼·��������
epoch = 1; % ����
% ��¼ÿ�غ����ų���
R_best = zeros(epochs, cities); %ÿ�ε�������Ⱥ������·��
L_best = inf .* ones(epochs, 1); % ÿ�ε�������Ⱥ����ѳ���
L_ave = zeros(epochs, 1); % ÿ�ε���ƽ������
% ��ʼ����
while epoch <= epochs
    % ���λ��
    %ants 50  ����ŵ� 40��������
    RandPos = []; 
    for i = 1: ceil(ants / cities)
        RandPos = [RandPos, randperm(cities)];
    end
    Route(:, 1) = (RandPos(1, 1:ants))';
    % ���50ֻ���ϰ�����pѡ����һ������
    for j = 2:cities
        for i = 1: ants
            Visited = Route(i, 1:j-1); %��¼���ʹ��ĳ��У���ֹ�ظ�
            NoVisited = zeros(1, (cities - j + 1)); % �����ʵĳ���
            P = NoVisited;  %�����ʵĳ���ѡ����ʷֲ�
            num = 1;
            for k = 1: cities
                if length(find(Visited == k)) == 0
                    NoVisited(num) = k;
                    num = num + 1; %ÿ����һ��������+1
                end
            end
          %�����ѡ���еĸ��ʷֲ�
            for k = 1: length(NoVisited) 
                P(k) = (Tau(Visited(end), NoVisited(k))^alpha) * (Eta(Visited(end), NoVisited(k))^beta);
            end
            P = P / sum(P);
               %������ԭ��ѡȡ��һ������
            Pcum = cumsum(P);  %cumsum��Ԫ���ۼӼ����
            select = find(Pcum >= rand);%������ĸ��ʴ���ԭ���ľ�ѡ������·��
            to_visit = NoVisited(select(1));
            Route(i, j) = to_visit;
        end
    end
       
    if epoch >= 2
        Route(1, :) = R_best(epoch - 1, :);
    end
    %%4.��¼���ε������·��
    Distance_epoch = zeros(ants, 1);
    for i = 1: ants
        R = Route(i, :);
        for j = 1: cities - 1
            Distance_epoch(i) = Distance_epoch(i) + Distance(R(j), R(j + 1)); %��ʼ����Ϊ0��m*1��������
        end
        Distance_epoch(i) = Distance_epoch(i) + Distance(R(1), R(cities)); %һ���������߹��ľ���
    end 
    L_best(epoch) = min(Distance_epoch); %��Ѿ���ȡ��С
    pos = find(Distance_epoch == L_best(epoch));
    R_best(epoch, :) = Route(pos(1), :);%���ֵ���������·��
    L_ave(epoch) = mean(Distance_epoch); %���ֵ������ƽ������
    epoch = epoch + 1;
    
    Delta_Tau = zeros(cities, cities); %��ʼʱ��Ϣ��Ϊn*n��0����
    for i = 1: ants
        for j = 1: (cities - 1)
            Delta_Tau(Route(i, j), Route(i, j + 1)) = Delta_Tau(Route(i, j), Route(i, j + 1)) + Q / Distance_epoch(i);
                 %�˴�ѭ����·����i��j���ϵ���Ϣ������
        end
        Delta_Tau(Route(i, 1), Route(i, cities)) = Delta_Tau(Route(i, 1), Route(i, cities)) + Q / Distance_epoch(i);
          %�˴�ѭ��������·���ϵ���Ϣ������
    end
    Tau = (1 - rho) .* Tau + Delta_Tau; %������Ϣ�ػӷ������º����Ϣ��
    Route = zeros(ants, cities);%ֱ������������
end
%% ���չʾ
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
% title('ƽ���������̾���');
