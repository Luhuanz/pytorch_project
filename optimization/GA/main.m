clear;clc;close all;
%%% 学习见下 
%%%https://blog.csdn.net/viafcccy/article/details/94429036?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166242797416781790750057%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166242797416781790750057&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-94429036-null-null.142^v46^pc_rank_34_2&utm_term=GA&spm=1018.2226.3001.4187
%%遗传参数设置
flow_a=60;%初始种群大小
i_l=-1; %问题解区间
i_r=2; %[-1,2] %问题解区间
lengths=22; %二进制编码长度
iters = 10000;%迭代次数
cros = 0.7;%杂交率
select_ratio = 0.5;%选择率
varS = 0.001;%变异率

%创建初始化种群 pop=[];
pop=[];
for i=1:flow_a
    pop(:,i)=i_l+(i_r-i_l)*rand;
end
pop_save=pop;
%绘制初始种群分布
x=linspace(-1,2,1000);  %（-1，2）区间上
y=m_Fun(x);  %建模的函数
plot(x,y);
hold on
for i=1:size(pop,2)   % size(pop)= 1 60  size(pop ,2)=60
    plot(pop(i),m_Fun(pop(i)),'ro');
end
hold off
title('创建初始种群');

%开始迭代
for time=1:iters
    %计算初始种群中每个个体的适应度
    fitness=m_Fitness(pop);
    %选择
    pop=m_Select(fitness,pop,select_ratio);
    %编码
    binpop=m_Coding(pop,lengths,i_l); %二进制
    %交叉
    kidsPop = crossover(binpop,flow_a,cros); %二进制    交叉 突变 都是基因层面的 用二进制表示
     %变异
    kidsPop = Variation(kidsPop,varS);%二进制
    %解码
    kidsPop=m_Incoding(kidsPop,i_l);  % 十进制 新的个体 不好的个体被杀死
    %更新种群
    pop=[pop kidsPop];
end
figure
x=linspace(-1,2,1000);
y=m_Fun(x);
plot(x,y);
hold on
for i=1:size(pop,2)
    plot(pop(i),m_Fun(pop(i)),'ro');
end
hold off
title('终止种群');

disp(['最优解：' num2str(max(m_Fun(pop)))]);
disp(['最大适应度：' num2str(max(m_Fitness(pop)))]);   
    
    


