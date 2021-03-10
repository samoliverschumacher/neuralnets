function [syn] = IniWeights( activtype , weightsize , varargin )
% Construct the nodes in each layer. First and Final Layer are input and output layers

if numel(varargin)>0
    KernelType = varargin{1};
else
    KernelType = 'Glorot';
    
end
NormDistType = false;
% Construct Synapses - initalise the weights
% For each function, a rule to select the range from which weights can be randomly initalised

switch KernelType
    case 'Zeros'
        
        syn = zeros( weightsize );
        
    case 'Ones'
        
        syn = ones( weightsize );
        
    case 'Glorot'
        sig_soft_gain = 1;
        if NormDistType
            ActFcnWIni_normal =[   { "tanh" , @(N_in, N_out) sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "sigmoid" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "softmax" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "leakyrelu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[-1 , 1] };...
                { "relu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[0 , 1] };...
                { "linear" , @(N_in, N_out) [-1 , 1] };  ];
            id_n = strcmpi( [ActFcnWIni_normal{:,1}] , activtype);
            mu = 0;
            sigma = abs(max(ActFcnWIni_normal{ id_n ,2}( weightsize(1) , weightsize(2) )));
            sz = weightsize;
            syn = normrnd(mu,sigma,sz);
        else
            ActFcnWIni =[   { "tanh" , @(N_in, N_out) sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "sigmoid" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "softmax" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "leakyrelu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[-1 , 1] };...
                { "relu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[0 , 1] };...
                { "linear" , @(N_in, N_out) [-1 , 1] };  ];
            
            id = strcmpi( [ActFcnWIni{:,1}] , activtype);
            
            RndWeights = rand( weightsize(1) ,  weightsize(2) ); % Uniformly Randomised weights
            iniWeightRange = ActFcnWIni{ id ,2}( weightsize(1) , weightsize(2) );
            syn = range(iniWeightRange ).*RndWeights - (range(iniWeightRange ))/2;
        end
        
    case 'orthogonal'
        
        NormDistType = true;
        % orthogonal matrix given by the QR decomposition of Z=QR for unit normal
        % distribution
        Z = randn( weightsize );
        
        [U,~,V] = svd(Z,'econ');
        if all(weightsize==size(U))
            syn = U;
        else
            syn = reshape(V,weightsize);
        end
        
    case 'He'
        
        sig_soft_gain = 4;
        if NormDistType
            ActFcnWIni_normal =[   { "tanh" , @(N_in, N_out) sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "sigmoid" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "softmax" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "leakyrelu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[-1 , 1] };...
                { "relu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[0 , 1] };...
                { "linear" , @(N_in, N_out) [-1 , 1] };  ];
            id_n = strcmpi( [ActFcnWIni_normal{:,1}] , activtype);
            mu = 0;
            sigma = abs(max(ActFcnWIni_normal{ id_n ,2}( weightsize(1) , weightsize(2) )));
            sz = weightsize;
            syn = normrnd(mu,sigma,sz);
        else
            ActFcnWIni =[   { "tanh" , @(N_in, N_out) sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "sigmoid" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "softmax" , @(N_in, N_out) [ sig_soft_gain ].*sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
                { "leakyrelu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[-1 , 1] };...
                { "relu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[0 , 1] };...
                { "linear" , @(N_in, N_out) [-1 , 1] };  ];
            
            id = strcmpi( [ActFcnWIni{:,1}] , activtype);
            
            RndWeights = rand( weightsize(1) ,  weightsize(2) ); % Uniformly Randomised weights
            iniWeightRange = ActFcnWIni{ id ,2}( weightsize(1) , weightsize(2) );
            syn = range(iniWeightRange ).*RndWeights - (range(iniWeightRange ))/2;
        end
        
end
    
% 
% if ~NormDistType
%     ActFcnWIni =[   { "tanh" , @(N_in, N_out) sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
%         { "sigmoid" , @(N_in, N_out) [ 1*Glorot_Xavier + 4*(~Glorot_Xavier) ].*sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
%         { "softmax" , @(N_in, N_out) [ 1*Glorot_Xavier + 4*(~Glorot_Xavier) ].*sqrt( 6 ./(N_in + N_out)).*[-1 , 1] }; ...
%         { "leakyrelu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[-1 , 1] };...
%         { "relu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[0 , 1] };...
%         { "linear" , @(N_in, N_out) [-1 , 1] };  ];
%     
%     id = strcmpi( [ActFcnWIni{:,1}] , activtype);
%     
%     RndWeights = rand( weightsize(1) ,  weightsize(2) ); % Uniformly Randomised weights
%     iniWeightRange = ActFcnWIni{ id ,2}( weightsize(1) , weightsize(2) );
%     syn = range(iniWeightRange ).*RndWeights - (range(iniWeightRange ))/2;
% else % use a normal distribution with STDDEV & MEAN=0
%     
%     ActFcnWIni_normal =[   { "tanh" , @(N_in, N_out) sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
%         { "sigmoid" , @(N_in, N_out) [ 1*Glorot_Xavier + 4*(~Glorot_Xavier) ].*sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
%         { "softmax" , @(N_in, N_out) [ 1*Glorot_Xavier + 4*(~Glorot_Xavier) ].*sqrt( 2 ./(N_in + N_out)).*[-1 , 1] }; ...
%         { "leakyrelu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[-1 , 1] };...
%         { "relu" , @(N_in, N_out) sqrt( 2 ./(N_in)).*[0 , 1] };...
%         { "linear" , @(N_in, N_out) [-1 , 1] };  ];
%     
%     id_n = strcmpi( [ActFcnWIni_normal{:,1}] , activtype);
%     
%     mu = 0;
%     sigma = abs(max(ActFcnWIni_normal{ id_n ,2}( weightsize(1) , weightsize(2) )));
%     sz = weightsize;
%     syn = normrnd(mu,sigma,sz);
% end




end