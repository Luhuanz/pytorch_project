%% 子函数
%
%题  目：Variation
%
%
%输   入：
%           pop              种群
%           VARIATIONRATE    变异率
%输   出：
%           pop              变异后的种群
%% 
function kidsPop = Variation(kidsPop,VARIATIONRATE)
for n=1:size(kidsPop,2)
    if rand<VARIATIONRATE
        temp = kidsPop{n};
        %找到变异位置
        location = ceil(length(temp)*rand);
        temp = [temp(1:location-1) num2str(~temp(location))...
            temp(location+1:end)];
       kidsPop{n} = temp;
    end
end