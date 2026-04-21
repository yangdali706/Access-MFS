function MacroF1 = Macro_F1( Pre_Labels,test_target)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
m1 = length(Pre_Labels(1,:));%列
m2 = length(Pre_Labels(:,1));%行
d = 0;
for i = 1:m2
    TP = 0;
    FP = 0;
    FN = 0;
    for j = 1:m1
        if test_target(i,j) == 1
            if Pre_Labels(i,j) ==1
                TP = TP +1;
            else
                FP = FP + 1;
            end
        else
           if Pre_Labels(i,j) ==1
               FN = FN +1;
           end  
        end 
    end
    SS = 0;
    if 2*TP+FP+FN~=0
        SS = 2*TP/(2*TP+FP+FN);
    end
    d = d + SS;
end
MacroF1 = d /m2;
end

