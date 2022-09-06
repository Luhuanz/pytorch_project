function pop=m_Incoding(binPop,irange_l)
%% 解码
popNum=1;
popNum = 1;%染色体包含的参数数量
for n=1:size(binPop,2)
    Matrix = binPop{1,n};
    for num=1:popNum
        pop(num,n) = bin2dec(Matrix);
    end
end
pop = pop./10^6+irange_l;
