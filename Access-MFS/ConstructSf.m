function [S_F,L_F,beta] = ConstructSf(F,num,option)
G=zeros(num.Label);
aa=sum(F.*F);
ab=2*F'*F;
for i=1:num.Label
    G(i,:)=aa(i)+aa-ab(i,:);
end
G=G';
[G,indG]=sort(G,'ascend');
S_F=zeros(num.Label);
for  col=1:num.Label
    row=indG(2:option.k_f+1,col);
    S_F(row,col)=(G(option.k_f+2,col)-G(2:option.k_f+1,col)) / (option.k_f*G(option.k_f+2,col)-sum(G(2:option.k_f+1,col))+eps);
end
beta=sum(0.5*(option.k_f*G(option.k_f+2,:)-sum(G(2:option.k_f+1,:))))/num.Label;
S_F=(S_F+S_F')/2;
D_F=diag(sum(S_F));
L_F=D_F-S_F;
end

