function MicroF1 =Micro_F1(Pre_Labels,test_target)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
[num_class,num_instance]=size(Pre_Labels);
TP = 0;
FP = 0;
FN = 0;
for i = 1:num_class
    for j = 1:num_instance
        if Pre_Labels(i,j) == 1
            if test_target(i,j) ==1
                TP = TP +1;
            else
                FP = FP + 1;
            end
        else
           if test_target(i,j) ==1
               FN = FN +1;
           end   
        end  
    end
end
if 2*TP+FP+FN~=0
    MicroF1 = 2*TP/(2*TP+FP+FN);
else
    MicroF1 = 0;
end
end

