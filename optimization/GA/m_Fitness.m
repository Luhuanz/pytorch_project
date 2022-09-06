function fitness=m_Fitness(pop)
%% Fitness Function  最小化 fitness
%y=xsin(3x)在[-1,2]上，最大值也不会超过2
%所以计算函数值到2的距离，距离最小时，即为最优解
%适应度函数为1/距离

%%%适应度函数用于评价某个染色体的适应度，用f(x)表示。有时需要区分染色体的适应度函数与问题的目标函数。
%%%例如：0-1背包问题的目标函数是所取得物品价值，但将物品价值作为染色体的适应度函数可能并不一定适合。
%%%适应度函数与目标函数是正相关的，可对目标函数作一些变形来得到适应度函数。

for n=1:size(pop,2)
    fitness(n)=1/(2-m_Fun(pop(:,n)));   %pop.shape 1 60  
    %fitness(n)=1/(10-m_Fun(pop(:,n))); %pop.shape 1,60
end

end
