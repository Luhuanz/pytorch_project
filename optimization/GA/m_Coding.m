function binPop=m_Coding(pop,pop_length,irange_l)
%% �����Ʊ��루����Ⱦɫ�壩
% ���룺pop--��Ⱥ
%      pop_length--���볤��
pop=round((pop-irange_l)*10^6);
for n=1:size(pop,2) %��ѭ��
    for k=1:size(pop,1) %��ѭ��
        dec2binpop{k,n}=dec2bin(pop(k,n));%dec2bin�����Ϊ�ַ�������
                                          %dec2binpop��cell����
        lengthpop=length(dec2binpop{k,n});
        for s=1:pop_length-lengthpop %����
            dec2binpop{k,n}=['0' dec2binpop{k,n}];
        end
    end
    binPop{n}=dec2binpop{k,n};   %ȡdec2binpop�ĵ�k��
end

    