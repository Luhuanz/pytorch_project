function pop=m_Incoding(binPop,irange_l)
%% ����
popNum=1;
popNum = 1;%Ⱦɫ������Ĳ�������
for n=1:size(binPop,2)
    Matrix = binPop{1,n};
    for num=1:popNum
        pop(num,n) = bin2dec(Matrix);
    end
end
pop = pop./10^6+irange_l;
