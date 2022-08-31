
%https://zhuanlan.zhihu.com/p/39993228

function [Q]=ahp(B)
%Q为权值，B为对比矩阵
%导入判别矩阵B
[n,m]=size(B);
%判别矩阵具有完全一致性
for i=1:n
    for j=1:n
        if B(i,j)*B(j,i)~=1   
        fprintf('i=%d,j=%d,B(i,j)=%d,B(j,i)=%d\n',i,j,B(i,j),B(j,i))  
        end  
    end
end
%求特征值特征向量,找到最大特征值对应的特征向量
[V,D]=eig(B);
tz=max(D);
tzz=max(tz);
c1=find(D(1,:)==max(tz));
tzx=V(:,c1);%特征向量
%权
quan=zeros(n,1);
for i=1:n
quan(i,1)=tzx(i,1)/sum(tzx);
end
Q=quan;
%一致性检验
CI=(tzz-n)/(n-1);
RI=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59];
%判断是否通过一致性检验
CR=CI/RI(1,n);
if CR>=0.1
   fprintf('没有通过一致性检验\n');
else
  fprintf('通过一致性检验\n');
end
 
end