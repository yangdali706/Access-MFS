function [S_X,L_X,alpha]=ConstructSx(W,X,F,num,option)

    M=zeros(num.Sample);
    aa1=sum((W'*X).*(W'*X));
    aa1=aa1';
    ab1=2*(W'*X)'*(W'*X);
    aa2=sum(F.*F,2);
    ab2=2*F*F';
    for i=1:num.Sample
        M(:,i)=aa1(i)+aa1-ab1(:,i)+1/2*(aa2(i)+aa2-ab2(:,i));
    end
    [M,indM]=sort(M,'ascend');
    S_X=zeros(num.Sample);
    
    for  col=1:num.Sample
        row=indM(2:option.k_s+1,col);
        S_X(row,col)=(M(option.k_s+2,col)-M(2:option.k_s+1,col)) / (option.k_s*M(option.k_s+2,col)-sum(M(2:option.k_s+1,col))+0.01);
    end
    alpha=sum(0.5*(option.k_s*M(option.k_s+2,:)-sum(M(2:option.k_s+1,:))))/num.Sample;
    S_X=(S_X+S_X')/2;
    D_X=diag(sum(S_X)); 
    L_X=D_X-S_X;

end


            