%% �Ӻ���
%
%��  Ŀ��Variation
%
%
%��   �룺
%           pop              ��Ⱥ
%           VARIATIONRATE    ������
%��   ����
%           pop              ��������Ⱥ
%% 
function kidsPop = Variation(kidsPop,VARIATIONRATE)
for n=1:size(kidsPop,2)
    if rand<VARIATIONRATE
        temp = kidsPop{n};
        %�ҵ�����λ��
        location = ceil(length(temp)*rand);
        temp = [temp(1:location-1) num2str(~temp(location))...
            temp(location+1:end)];
       kidsPop{n} = temp;
    end
end