function binPop=m_Coding(pop,pop_length,irange_l)
%% 二进制编码（生成染色体）
% 输入：pop--种群
%      pop_length--编码长度
pop=round((pop-irange_l)*10^6);
for n=1:size(pop,2) %列循环
    for k=1:size(pop,1) %行循环
        dec2binpop{k,n}=dec2bin(pop(k,n));%dec2bin的输出为字符向量；
                                          %dec2binpop是cell数组
        lengthpop=length(dec2binpop{k,n});
        for s=1:pop_length-lengthpop %补零
            dec2binpop{k,n}=['0' dec2binpop{k,n}];
        end
    end
    binPop{n}=dec2binpop{k,n};   %取dec2binpop的第k行
end

    