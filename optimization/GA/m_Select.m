function parentPop=m_Select(matrixFitness,pop,SELECTRATE)
%% ѡ��
% ���룺matrixFitness--��Ӧ�Ⱦ���
%      pop--��ʼ��Ⱥ
%      SELECTRATE--ѡ����

sumFitness=sum(matrixFitness(:));%����������Ⱥ����Ӧ��

accP=cumsum(matrixFitness/sumFitness);%�ۻ�����
%���̶�ѡ���㷨
for n=1:round(SELECTRATE*size(pop,2))
    matrix=find(accP>rand); %�ҵ������������ۻ�����
    if isempty(matrix)
        continue
    end
    parentPop(:,n)=pop(:,matrix(1));%���׸������������ۻ����ʵ�λ�õĸ����Ŵ���ȥ
end
end