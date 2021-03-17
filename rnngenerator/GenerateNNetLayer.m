function [NNLayer] = GenerateNNetLayer( nU , batchsize , priorNunits , Type , ActvFcn , varargin )

% Neuron Activation Functions
% fcn_types = struct('tanh',1,'sigmoid',2 ,'softmax',3,'leakyrelu',4,'relu',5,'linear',6);
%% Basic Dense layer

sz_ij = [priorNunits , nU];
sz_ii = [nU , nU];

% Initialise the weights
[synij] = IniWeights( ActvFcn , sz_ij , 'Glorot' );
[synb] = IniWeights( ActvFcn , [1 , nU] , 'Zeros' );

NNLayer_Dense = struct( 'Type' ,"dense" ,'ActFcn', ActvFcn , 'Nunits', nU , 'XInput' , zeros( batchsize , priorNunits ) , ...
    'Activations', struct('HiddenIn', zeros(batchsize , nU), 'HiddenOut', zeros(batchsize , nU)) , ...
    'Weights', struct('wIN', synij ,'wb', synb ) , ...
    'BP_pOut', struct('wIN',{repmat( zeros(sz_ij) ,[1 1 2])} ,'wb', {repmat( zeros([1 , nU]) ,[1 1 2])} ) , ...
    'dEdW', struct('wIN', zeros(sz_ij) ,'wb', zeros([1 , nU]) ) , ...
    'Connectivity' , ones( sz_ij ) );

switch Type
    case 'LSTM'
            
    NNLayer_LSTM = NNLayer_Dense;
    NNLayer_LSTM.('Type') = "LSTM";
    %% LSTM Layer
    nTimesteps = varargin{1};
    NNLayer_LSTM.predictsequence = false;
    NNLayer_LSTM.PropBatchinSequence = false;
    NNLayer_LSTM.resetstate = true;
    NNLayer_LSTM.SelfReferencing = false;
    NNLayer_LSTM.TeacherForcing = false;
    NNLayer_LSTM.Peephole = false;
    NNLayer_LSTM.Attention = false;
    NNLayer_LSTM.BiLSTM = false;
    NNLayer_LSTM.InputMask.hasMask = false;
    
    if numel( varargin )>1
        AdditionalSettings = varargin{2};
        SettingNames = fieldnames(AdditionalSettings);
        for fn = 1:numel(SettingNames)
            NNLayer_LSTM.(SettingNames{fn}) = AdditionalSettings.(SettingNames{fn});
        end
    end
    
    if NNLayer_LSTM.BiLSTM==true
        AdditionalSettings.BiLSTM = false;
        [NNLayer_BiLSTM] = GenerateNNetLayer( nU , batchsize , priorNunits , "BiLSTM" , ActvFcn , nTimesteps , AdditionalSettings );
        NNLayer_BiLSTM.Type="LSTM";
        NNLayer_BiLSTM.BiLSTM = true;
        NNLayer_BiLSTM.curStep = 1;
        NNLayer_BiLSTM.InputMask = NNLayer_BiLSTM.Forward.InputMask;
        NNLayer_BiLSTM.resetstate = NNLayer_BiLSTM.Forward.resetstate;
        NNLayer_BiLSTM.PropBatchinSequence = NNLayer_BiLSTM.Forward.PropBatchinSequence;
        NNLayer_BiLSTM.MergeFcn = NNLayer_BiLSTM.Forward.MergeFcn;
        NNLayer_BiLSTM.SelfReferencing = NNLayer_BiLSTM.Forward.SelfReferencing;
        
        NNLayer = NNLayer_BiLSTM;
    else
    
    NNLayer_LSTM.InputMask.Mask = zeros( batchsize , nTimesteps );
    
    if NNLayer_LSTM.TeacherForcing
        NNLayer_LSTM.TeacherForcedTarget = zeros( [batchsize , nTimesteps , nU] );
%         NNLayer_LSTM.teacherforcingratio = 0.5; %1=always teacher force, 0=never
    end
    if NNLayer_LSTM.SelfReferencing
        % In self referencing, Xinput is the prior timestep Hidden Output,
        % so a layer has to be evaluated once, at each timestep, rather
        % than in an unbroken loop.
        NNLayer_LSTM.curStep = 1;
    end
    NNLayer_LSTM.curStep = 1;
    
    %     1-tanh    2-sigmoid    3-softmax    4-leakyrelu    5-relu    6-linear
    NNLayer_LSTM.('ActFcn') = "tanh";
    NNLayer_LSTM.Activations.('HiddenOut') = zeros( batchsize , nTimesteps , nU );
    NNLayer_LSTM.Activations.('HiddenIn') = zeros( batchsize , nTimesteps , nU );
    
    NNLayer_LSTM.XInput = zeros( batchsize , nTimesteps , priorNunits );
    NNLayer_LSTM.Activations.('MemoryIn') = zeros( batchsize , [1] , nU );
    
    
    % NNLayer_LSTM.Weights = rmfield( NNLayer_LSTM.Weights , 'wIN' );
    NNLayer_LSTM.Nstates = nTimesteps;
    NNLayer_LSTM.Activations.Memory = zeros( batchsize , nTimesteps , nU );
    NNLayer_LSTM.Activations.Gates = struct( 'forget' , NNLayer_LSTM.Activations.Memory ,'input', NNLayer_LSTM.Activations.Memory ,'output', NNLayer_LSTM.Activations.Memory ,'activate', NNLayer_LSTM.Activations.Memory );
    NNLayer_LSTM.Activations.('GatesIn') = struct( 'forget' , NNLayer_LSTM.Activations.Memory ,'input', NNLayer_LSTM.Activations.Memory ,'output', NNLayer_LSTM.Activations.Memory ,'activate', NNLayer_LSTM.Activations.Memory );
    NNLayer_LSTM.Inistates = struct('Hidden',NNLayer_LSTM.Activations.HiddenOut(:,1,:) ,'Cell',NNLayer_LSTM.Activations.Memory(:,1,:) );
    
    
    % Initialise the weights
    synij_sigmoid = IniWeights( "sigmoid" , sz_ij , 'Glorot' );
    synij_tanh = IniWeights( "tanh" , sz_ij , 'Glorot' );
	synii_sigmoid = IniWeights( "sigmoid" , sz_ii , 'orthogonal' );
    synii_tanh = IniWeights( "tanh" , sz_ii , 'orthogonal' );
	synb_sigmoid = IniWeights( "sigmoid" , [1 , nU] , 'Zeros' );
    synb_tanh = IniWeights( "tanh" , [1 , nU] , 'Zeros' );
    
    NNLayer_LSTM.Weights = rmfield( NNLayer_LSTM.Weights , {'wIN','wb'} );
    NNLayer_LSTM.Weights.wIN = struct( 'forget' , synij_sigmoid ,'input', synij_sigmoid ,'output', synij_sigmoid ,'activate', synij_tanh );
    NNLayer_LSTM.Weights.wrec = struct( 'forget' , synii_sigmoid ,'input', synii_sigmoid ,'output', synii_sigmoid ,'activate', synii_tanh );
    NNLayer_LSTM.Weights.wb = struct( 'forget' , ones( size(synb_sigmoid) ) ,'input', synb_sigmoid ,'output', synb_sigmoid ,'activate', synb_tanh );
    
    NNLayer_LSTM.Weights.wpeep = struct( 'forget' , synii_sigmoid ,'input', synii_sigmoid ,'output', synii_sigmoid );
    
    % Back propogation & Optimizer variables.
    NNLayer_LSTM.dEdW = struct('wIN', struct('forget',zeros([ sz_ij , batchsize ]),'input',zeros([ sz_ij , batchsize ]) ,'activate',zeros([ sz_ij , batchsize ]) ,'output',zeros([ sz_ij , batchsize ]) ) ...
        ,'wrec', struct('forget',zeros([ sz_ii , batchsize ]) ,'input',zeros([ sz_ii , batchsize ]) ,'activate',zeros([ sz_ii , batchsize ]) ,'output',zeros([ sz_ii , batchsize ]) ) ...
        ,'wb', struct('forget',zeros([1 , nU, batchsize ]) ,'input',zeros([1 , nU, batchsize ]) ,'activate',zeros([1 , nU, batchsize ]) ,'output',zeros([1 , nU, batchsize ]) ) ...
        ,'wpeep', struct('forget',zeros([ sz_ii , batchsize ]) ,'input',zeros([ sz_ii , batchsize ]) ,'output',zeros([ sz_ii , batchsize ]) ) );
    NNLayer_LSTM.BP_pOut = struct('wIN', struct('forget', repmat( zeros(sz_ij) ,[1 1 2]) ,'input', repmat( zeros(sz_ij) ,[1 1 2]) ,'activate', repmat( zeros(sz_ij) ,[1 1 2]) ,'output', repmat( zeros(sz_ij) ,[1 1 2]) ) ...
        ,'wrec', struct('forget', repmat( zeros(sz_ii) ,[1 1 2])  ,'input', repmat( zeros(sz_ii) ,[1 1 2]) ,'activate', repmat( zeros(sz_ii) ,[1 1 2]) ,'output', repmat( zeros(sz_ii) ,[1 1 2]) ) ...
        ,'wb', struct('forget', repmat( zeros([1 , nU]) ,[1 1 2]) ,'input', repmat( zeros([1 , nU]) ,[1 1 2]) ,'activate', repmat( zeros([1 , nU]) ,[1 1 2]) ,'output', repmat( zeros([1 , nU]) ,[1 1 2]) ) ...
        ,'wpeep', struct('forget', repmat( zeros(sz_ii) ,[1 1 2]) ,'input', repmat( zeros(sz_ii) ,[1 1 2]) ,'output', repmat( zeros(sz_ii) ,[1 1 2]) ) );
    
    
    NNLayer = NNLayer_LSTM;
    
    end
    
	case 'LSTMattention'
    %% LSTM Layer WITH Attention


    case 'BiLSTM'
    
        nTimesteps = varargin{1};
        if numel(varargin)>1
            AdditionalSettings = varargin{2};
        end
        if rem(nU,2)~=0
            disp('Must have even number of units for each direction of Bidirectional LSTM to split evenly')
        end
        [NNLayer_BiLSTM.Forward] = GenerateNNetLayer( nU./2 , batchsize , priorNunits , "LSTM" , ActvFcn , nTimesteps , AdditionalSettings );
        [NNLayer_BiLSTM.Backward] = GenerateNNetLayer( nU./2 , batchsize , priorNunits , "LSTM" , ActvFcn , nTimesteps , AdditionalSettings );
        
        NNLayer_BiLSTM.Type ="LSTM";
        
        NNLayer_BiLSTM.BiCellOut = cat( 3 , NNLayer_BiLSTM.Forward.Activations.Memory , NNLayer_BiLSTM.Backward.Activations.Memory );
        NNLayer_BiLSTM.BiHout = cat( 3 , NNLayer_BiLSTM.Forward.Activations.HiddenOut , NNLayer_BiLSTM.Backward.Activations.HiddenOut );
        
        NNLayer_BiLSTM.XInput = NNLayer_BiLSTM.Forward.XInput;
        NNLayer_BiLSTM.Activations.HiddenOut = cat( 3 , NNLayer_BiLSTM.Forward.Activations.HiddenOut , NNLayer_BiLSTM.Backward.Activations.HiddenOut);
        NNLayer_BiLSTM.Activations.Memory = cat( 3 , NNLayer_BiLSTM.Forward.Activations.Memory , NNLayer_BiLSTM.Backward.Activations.Memory );
        NNLayer_BiLSTM.Inistates.Cell = cat( 3 , NNLayer_BiLSTM.Forward.Inistates.Cell  , NNLayer_BiLSTM.Backward.Inistates.Cell );
        NNLayer_BiLSTM.Inistates.Hidden = cat( 3 , NNLayer_BiLSTM.Forward.Inistates.Hidden  , NNLayer_BiLSTM.Backward.Inistates.Hidden );
        NNLayer_BiLSTM.predictsequence = true;
        NNLayer_BiLSTM.Nstates = nTimesteps;
        NNLayer_BiLSTM.Nunits = nU;
        
        
        NNLayer = NNLayer_BiLSTM;
        
    otherwise
    
    NNLayer = NNLayer_Dense;
end


end


