function parentPop=m_Select(matrixFitness,pop,SELECTRATE)
%% 选择
% 输入：matrixFitness--适应度矩阵
%      pop--初始种群
%      SELECTRATE--选择率

sumFitness=sum(matrixFitness(:));%计算所有种群的适应度

accP=cumsum(matrixFitness/sumFitness);%累积概率
%轮盘赌选择算法
for n=1:round(SELECTRATE*size(pop,2))
    matrix=find(accP>rand); %找到比随机数大的累积概率
    if isempty(matrix)
        continue
    end
    parentPop(:,n)=pop(:,matrix(1));%将首个比随机数大的累积概率的位置的个体遗传下去
end
end