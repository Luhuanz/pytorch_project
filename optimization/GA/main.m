clear;clc;close all;
%%% ѧϰ���� 
%%%https://blog.csdn.net/viafcccy/article/details/94429036?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166242797416781790750057%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166242797416781790750057&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-94429036-null-null.142^v46^pc_rank_34_2&utm_term=GA&spm=1018.2226.3001.4187
%%�Ŵ���������
flow_a=60;%��ʼ��Ⱥ��С
i_l=-1; %���������
i_r=2; %[-1,2] %���������
lengths=22; %�����Ʊ��볤��
iters = 10000;%��������
cros = 0.7;%�ӽ���
select_ratio = 0.5;%ѡ����
varS = 0.001;%������

%������ʼ����Ⱥ pop=[];
pop=[];
for i=1:flow_a
    pop(:,i)=i_l+(i_r-i_l)*rand;
end
pop_save=pop;
%���Ƴ�ʼ��Ⱥ�ֲ�
x=linspace(-1,2,1000);  %��-1��2��������
y=m_Fun(x);  %��ģ�ĺ���
plot(x,y);
hold on
for i=1:size(pop,2)   % size(pop)= 1 60  size(pop ,2)=60
    plot(pop(i),m_Fun(pop(i)),'ro');
end
hold off
title('������ʼ��Ⱥ');

%��ʼ����
for time=1:iters
    %�����ʼ��Ⱥ��ÿ���������Ӧ��
    fitness=m_Fitness(pop);
    %ѡ��
    pop=m_Select(fitness,pop,select_ratio);
    %����
    binpop=m_Coding(pop,lengths,i_l); %������
    %����
    kidsPop = crossover(binpop,flow_a,cros); %������    ���� ͻ�� ���ǻ������� �ö����Ʊ�ʾ
     %����
    kidsPop = Variation(kidsPop,varS);%������
    %����
    kidsPop=m_Incoding(kidsPop,i_l);  % ʮ���� �µĸ��� ���õĸ��屻ɱ��
    %������Ⱥ
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
title('��ֹ��Ⱥ');

disp(['���Ž⣺' num2str(max(m_Fun(pop)))]);
disp(['�����Ӧ�ȣ�' num2str(max(m_Fitness(pop)))]);   
    
    


