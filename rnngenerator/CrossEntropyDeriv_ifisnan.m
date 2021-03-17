function [result_otherwise] = CrossEntropyDeriv_ifisnan( Real, Pred )
% if Real value and Predicted value are equal due to numerical specificity of matlab, 
% Cross Entropy Loss derivative leads to a NaN, because 0/0 = NaN.

equalid = (Real==Pred);

equalones = equalid & Real==1;

equalzeros = equalid & Real==0;

result_otherwise = (-Real./Pred) + ( (1-Real)./(1-Pred) );

result_otherwise(equalones) = -1;

result_otherwise(equalzeros) = 1;
% 
% if ( Real==Pred ) && Real==1
%     result_otherwise = -1;
% elseif ( Real==Pred ) && Real==0
%     result_otherwise = 1;
% else
%     result_otherwise = (-Real./Pred) + ( (1-Real)./(1-Pred) );
% end
    




% Logits = [1 1 8];
% 
% Pred_sft = Actvfcn( Logits , false , 3 )
% 
% Real = [0 0 1];
% 
% dXent = CrossEntropyDeriv_ifisnan( Real, Pred_sft )
% 
% 
% dsft = Actvfcn( Logits , true , 3 )*dXent'
% 
% Pred_sft - Real