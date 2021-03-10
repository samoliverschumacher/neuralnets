function [outV] = Actvfcn( inV , deriv , act_type )
% outV = NaN(size(inV));
% fcn_types.tanh = 1;
% fcn_types.sigmoid = 2;
% fcn_types.softmax = 3;
% fcn_types.leakyrelu = 4;
% fcn_types.relu = 5;
% fcn_types.linear = 6;

switch act_type
    case 2%'sigmoid' % nonlin_1 = @(x,deriv) strcmp(deriv,'true').*((1./(1 + exp(-x))).*( 1 - (1./(1 + exp(-x)))))  + strcmp(deriv,'false').*(1./(1 + exp(-x)));
        outV = (1./(1 + exp(-inV)+10e-8));
        if ~deriv
        else % compute derivative of activation function
            outV = ( outV .*( 1 - outV));
        end
        
    case 1%'tanh'
        if ~deriv
            %             norm_const = -max(inV);
            outV = tanh(inV);%exp(inV+norm_const)/sum(exp(inV+norm_const));
        else % compute derivative of activation function
            %             norm_const = max(inV);
            outV = 1 - tanh(inV).^2;
        end
        
    case 4%'leakyrelu' % nonlin_1 = @(x,deriv) strcmp(deriv,'true').*((x>0).*1 + (x<=0).*0) + strcmp(deriv,'false').*max(0,x);
        if ~deriv
            outV = max(0.1.*inV,inV);
        else % compute derivative of activation function
            outV = ((inV>0) + (inV<=0).*0.1);
        end
        
    case 5%'relu' % nonlin_1 = @(x,deriv) strcmp(deriv,'true').*((x>0).*1 + (x<=0).*0) + strcmp(deriv,'false').*max(0,x);
        if ~deriv
            outV = max(0,inV);
        else % compute derivative of activation function
            outV = ((inV>0) + (inV<=0).*0);
        end
        
    case 3%'softmax'
        if ~deriv
            norm_const = -max(inV , [] , 2); % [ Batches x Logit scores (features) ]
            Z = inV+norm_const;
            Z = inV-max(inV,[],2);
            outV = exp(Z) ./ sum( exp(Z) , 2 );%             outV = sinh(inV) ./ cosh(inV);
        else % compute derivative of activation function
            norm_const = -max(inV , [] , 2);
            Z = inV+norm_const;
            Z = inV-max(inV,[],2);
            softmax = exp( Z ) ./ sum( exp( Z ) , 2);%             outV = sinh(inV) ./ cosh(inV);
            outV = diag(softmax) - repmat( softmax' , 1 , numel(inV) ).*repmat( softmax , numel(inV) , 1 );
            %
            %             for ii=1:size(inV,2) % i-th outputs,
            %                 for jj=1:size(inV,2) % j-th inputs
            %                     if ii==jj
            %                         dSoft(ii,jj) = softmax(ii).*(1-softmax(jj)); % partial deriv of the ith output wrt jth input
            %                     else
            %                         dSoft(ii,jj) = - softmax(ii).*softmax(jj) ;
            %                     end
            %                 end
            %             end
        end
        
    case 6%'linear' % nonlin_1 = @(x,deriv) strcmp(deriv,'true').*((x>0).*1 + (x<=0).*0) + strcmp(deriv,'false').*max(0,x);
        if ~deriv
            outV = inV;
        else % compute derivative of activation function
            outV = 1;
        end
        
end
% if any(isnan(outV)) , keyboard , end
end