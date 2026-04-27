function selfea_index= Access_MFS(X,Y,option)
%%
num.Sample=size(X,2);
num.Label=size(Y,2);
num.Feature=size(X,1);
I_n=eye(num.Sample);
one_n=ones(num.Sample,1);
H=I_n-(1/num.Sample)*(one_n*one_n');
U=eye(num.Sample);
for item=1:num.Sample
    if max(Y(item,:))==1
        U(item,item)=1;
    else
        U(item,item)=0; 
    end
end
%Initialize variables
F=Y; 
W=rand(num.Feature,num.Label);
D_w=eye(num.Feature);
[S_X,L_X,alpha]=ConstructSx(W,X,F,num,option);
[S_F,L_F,beta]=ConstructSf(F,num,option);
obj_value=1;
obj_valueopt=log(trace((H*X'*W-H*F)'*(H*X'*W-H*F))+option.lambda*trace(W'*D_w*W)...
    +trace((F-Y)'*U*(F-Y))+1/2*option.theta*trace(F'*L_X*F) ...
    +option.theta*trace(W'*X*L_X*X'*W)+option.theta*alpha*sum(sum(S_X.*S_X))...
    +option.mu*trace(F*L_F*F')+option.mu*beta*sum(sum(S_F.*S_F)));
Error_obj=obj_valueopt-obj_value;

while abs(Error_obj)>option.stopObj

    obj_value=obj_valueopt;

    %Update W
    Rt = X*H*X'+option.lambda.*D_w+option.theta*X*L_X*X';
    B=Rt^(-1/2)*X*H*F;
    [B_U,~,B_V]=svd(B,'econ');
    A=B_U*B_V';
    W=Rt^(-1/2)*A;
    tmp=2*((sum(W.*W,2)+eps).^(0.5));
    D_w=diag(1./tmp);
    %Update F
    Q=H+option.theta* L_X+U;
    C=-(H*X'*W+U*Y);
    F=lyap(Q,L_F,C);
    %Update S_X and S_F
    [S_X,L_X,alpha]=ConstructSx(W,X,F,num,option);
    [S_F,L_F,beta]=ConstructSf(F,num,option);
    
    obj_valueopt=log(trace((H*X'*W-H*F)'*(H*X'*W-H*F))+option.lambda*trace(W'*D_w*W)...
    +trace((F-Y)'*U*(F-Y))+1/2*option.theta*trace(F'*L_X*F) ...
    +option.theta*trace(W'*X*L_X*X'*W)+option.theta*alpha*sum(sum(S_X.*S_X))...
    +option.mu*trace(F*L_F*F')+option.mu*beta*sum(sum(S_F.*S_F)));
    
    Error_obj=obj_valueopt-obj_value;
end

%Select optimal features
L2_norm=sum(W.*W,2);
[~,Index]=sort(L2_norm,'descend');
selfea_index=Index(1:option.sel_fea);

end

