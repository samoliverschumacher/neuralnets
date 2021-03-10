function    [NNModels, Prediction, ErrorMetric1, ErrorMetric2, varargout] = TrainSequnceNNetV3(...
    NNModels, epochs, trainingSettings , DataX_batched, DataY_batched, InputMask, OutputMask, varargin)
tic

Nlayers = numel(NNModels);
batchsize = size(DataX_batched{1},1);

if numel(varargin) > 0
    predictonly = varargin{1};
    if numel(varargin) > 1
        displaytrainingprogress = varargin{2};
        if numel(varargin) > 2
            DataX_validation = varargin{3}{1};
            DataY_validation = varargin{3}{2};
        end
    else
          displaytrainingprogress = false;
    end
else
    predictonly = false;
    displaytrainingprogress = true;
end

[NetworkSettings.gradclip  ,gradclip] = deal(trainingSettings.gradclip);
GDOptimizer = trainingSettings.GDOptimizer;
learnrate = trainingSettings.learnrate;
LossType = trainingSettings.LossType;
if gradclip
      if displaytrainingprogress
      disp('Gradient Clipping is ON') , end
else
      if displaytrainingprogress , disp('Gradient Clipping is OFF') , end
end



[Nbatches, OptParams, lrsched, classificationproblem, LossFcn, Loss_delta, OutActiv , dCda, ErrorMetric1 , ErrorMetric2, Prediction, sampleidx, gradnormalisation] ...
    = initialiseNNet(NNModels,DataX_batched,DataY_batched, GDOptimizer, LossType, epochs, learnrate );
gradnorm = @(V) sqrt(sum( V.^2));


NetworkSettings.LossFcn = LossFcn;
NetworkSettings.Loss_delta = Loss_delta;

plottingiterations = false;%2==numel(NNModels);




LSTMType = contains( cellfun(@(K) K.Type, NNModels), 'LSTM' );
DenseType = contains( cellfun(@(K) K.Type, NNModels), 'dense' );

[dt] = find(DenseType);
[lt] = find(LSTMType);

fcn_types = struct('tanh',1,'sigmoid',2 ,'softmax',3,'leakyrelu',4,'relu',5,'linear',6);

RequiresTimeDistributedWrapper = false(size(NNModels));
for ci = 1:numel(dt)
    lstmb4dense = lt(((lt)==(dt(ci)-1)));
    if ~isempty(lstmb4dense) % if a LSTM comes after a dense layer
        if LSTMType( lstmb4dense )==true && DenseType( dt(ci) )==true && NNModels{lstmb4dense}.predictsequence==true
            RequiresTimeDistributedWrapper(dt(ci)) = true;
        end
    end
end

%% initialise training variables

% set up cell state initialisation, and prior timestep hidden output to
% feed into first timestep of each sample sequence in each batch
Ini_PriorEpochThisbatch = cell( Nbatches , Nlayers);
Ini_LastBatch = cell(1, Nlayers);
StoreLayerInputs = cell(1, Nlayers);
for lay=1:Nlayers
    StoreLayerInputs{lay} = NNModels{lay}.XInput;
end
SavedStates = cell(1, Nlayers);

seedStateFromPriorLayer = false;
NetworkSettings.seedStateFromPriorLayer = seedStateFromPriorLayer;

for lay = lt
    inisz{lay} = zeros(size(NNModels{lay}.Activations.HiddenOut ));
    if predictonly == true
        Ini_LastBatch{1,lay} = NNModels{(lay)}.Inistates;% struct('Hidden',inisz,'Cell',inisz);
        Ini_PriorEpochThisbatch(:,lay) = repmat( { NNModels{(lay)}.Inistates }, Nbatches , 1 );
        SavedStates{lay} = struct( 'Inistates', struct('Hidden',inisz{lay},'Cell',inisz{lay}) );
        
    else
        Ini_LastBatch{1,lay} = struct('Hidden',inisz{lay},'Cell',inisz{lay});
        Ini_PriorEpochThisbatch(:,lay) = repmat( {struct('Hidden',inisz{lay},'Cell',inisz{lay})}, Nbatches , 1 );
    end
end



UseOutputerrorMasking = ~isempty(OutputMask);
UseInputMasking = ~isempty(InputMask) & any(arrayfun(@(ll) NNModels{ll}.InputMask.hasMask,lt));

if classificationproblem
    Padding = DataY_batched{1}(:,1,:).*0;
    Padding( : , 1 , end ) = 1;% Padding one-hot variable is always in the last place in the OH vector
    if UseOutputerrorMasking
        LossFcn = @(Pred,Real)-nansum(Real.*log(Pred),3);
    end
end

%% Establish layer to layer processes
LayerOperation = [repmat( "regular" , 1 , Nlayers); cellfun(@(x) x.Type, NNModels)];
TDistOut = repmat( {[]} , 1 , Nlayers);

Emb_dEdW = cell(1, Nlayers);
if size(DataX_batched{1},3)>1 && NNModels{1}.Type=="dense" % first layer may be an embedding layer unattached to a LSTM
    lay = 1;
    LayerOperation(1,lay) = "SequenceEmbedding";
    SeqEmbOut{lay} = zeros( [size(NNModels{lay}.Activations.HiddenOut) , size(DataX_batched{lay},2)]);
    Emb_dEdW{lay}.dEdW = NNModels{lay}.dEdW;
    Emb_dEdW{lay}.InitilaiseValues = Emb_dEdW{lay}.dEdW;
end

for lay=2:Nlayers
    LayerOperation(2,lay) = NNModels{lay}.Type;
    if contains(NNModels{lay-1}.Type,["LSTM","BiLSTM"]) && NNModels{lay}.Type=="LSTM" % LSTM to LSTM
        if NNModels{lay}.SelfReferencing
            LayerOperation(1,lay) = "SelfReferencing";
            if ~isempty(NNModels{lay}.EmbeddingLayer)
                Emb_dEdW{lay} = NNModels{lay}.EmbeddingLayer.dEdW;
                Emb_dEdW{lay}.InitilaiseValues = Emb_dEdW{lay};
            end
            Prj_dEdW = NNModels{lay}.ProjectionLayer.dEdW;
            Prj_dEdW.InitilaiseValues = Prj_dEdW;
            
        elseif NNModels{lay-1}.predictsequence == false
            LayerOperation(1,lay) = "RepeatVector";
        elseif NNModels{lay-1}.predictsequence == true % All to All
            LayerOperation(1,lay) = "regular";
        end
        
    elseif contains(NNModels{lay-1}.Type,["LSTM","BiLSTM"]) && NNModels{lay}.Type=="dense" % LSTM to Dense
        
        if NNModels{lay-1}.predictsequence == false
            LayerOperation(1,lay) = "regular";
        elseif NNModels{lay-1}.predictsequence == true
            LayerOperation(1,lay) = "TimeDistributed";
            
            if NNModels{end-1}.Type=="LSTM"
                Ntimesteps = NNModels{end-1}.Nstates;
                if NNModels{end-1}.predictsequence
                    Prd_ti = [Ntimesteps:-1:1];
                    TDEUini = NNModels{end}.dEdW;
                    TimeDistributedErrorUpdates = TDEUini;
                else
                    Prd_ti = Ntimesteps; % Hfinal = NNLayer2.Activations.HiddenOut(:,end,:);
                    TDEUini = NNModels{end}.dEdW;
                    TimeDistributedErrorUpdates = TDEUini;
                end
            end
            
        end
        
    elseif NNModels{lay-1}.Type=="dense" && NNModels{lay}.Type=="dense" % Dense to Dense
        
        LayerOperation(1,lay) = "regular";
        
    elseif NNModels{lay-1}.Type=="dense" && NNModels{lay}.Type=="LSTM" % Dense to LSTM
        
        LayerOperation(1,lay) = "regular"; % LayerOperation(1,lay) = "SequenceEmbedding";
        
    end
end

NetworkSettings.LayerOperation  =LayerOperation;

if numel(size(DataX_batched{1}))>3
    NewDataperEpoch = true;
else
    NewDataperEpoch= false;
end
NetworkSettings.NewDataperEpoch = NewDataperEpoch;

%% Display training / Prediction settings


if predictonly    
else
%       function printNNetsettings( NetworkSettings )
	LogicalString = {'False','True'};
    disp('    Network Settings:')
    setting_names = fieldnames(NetworkSettings);
    numchar = max(cellfun(@numel ,setting_names )') - cellfun(@numel ,setting_names )' ;
    for cc=1:numel(setting_names), setting_names{cc,2} = [repmat(' ',[1,numchar(cc)]), setting_names{cc},' : ']; end
    
    for settingId = 1:numel(setting_names(:,1))
        switch class( NetworkSettings.(setting_names{settingId}) )
            case 'logical'
                fprintf([setting_names{settingId,2}, ' : %s \n'],LogicalString{NetworkSettings.(setting_names{settingId})+1})
            case 'function_handle'
                fprintf([setting_names{settingId,2}, ' : %s \n'],func2str(NetworkSettings.(setting_names{settingId})))
                
            case 'string'
                disp(setting_names{settingId,2}), disp(NetworkSettings.(setting_names{settingId}))
            otherwise
                disp(setting_names{settingId,2} ) , disp(NetworkSettings.(setting_names{settingId}) )
        end
    end
%       end
end

%% loop over batches to Train or Predict
% disp("initial cell: " +NNModels{lt(end)}.Inistates.Cell); disp("initial hidden: " +NNModels{lt(end)}.Inistates.Hidden); disp("NN cell: " +NNModels{lt(end)}.Activations.Memory); disp("NN hidden: " +NNModels{lt(end)}.Activations.HiddenOut);
for it = 1:epochs
    for btc = 1:size(DataX_batched,1)
        %% Select Data observations
        if NewDataperEpoch
            X_input = DataX_batched{btc,1}(:,:,:,it);
            Y_target = DataY_batched{btc,1}(:,:,:,it);
        else
            X_input = DataX_batched{btc,1}(:,:,:);
            Y_target = DataY_batched{btc,1}(:,:,:);
        end
        
        %% FORWARD PROPOGATE
        for lay = 1:Nlayers

                
            if lay==1
                Xin_k = X_input;
            else
                
                % Format Input to Layer, based on Layer operation.
                switch LayerOperation(1,lay)
                    case 'SequenceEmbedding' % 'dense' to 'LSTM'
                        SeqEmbOut{lay} = zeros( [size(NNModels{lay}.Activations.HiddenOut) , size(Xin_k,2)]);
                    case 'RepeatVector' % LSTM to LSTM
                        Xin_k = repmat( NNModels{lay-1}.Activations.HiddenOut(:, end, :) , [1 , size(NNModels{lay}.Activations.HiddenOut,2) , 1] );
                    case 'SelfReferencing' % LSTM to LSTM to selfLSTM - UNDEVELOPED
                        %                         Xin_k = NNModels{lay-1}.Activations.HiddenOut(:, end, :);
                        InitialiserToken = Y_target(:,1,:);
                        Xin_k = zeros( size(Y_target) );
                        Xin_k(:,1,:) = InitialiserToken; % self referencing decoder takes the "Start of String" initialiser in the first timestep of the Y target
                        if NNModels{lay}.TeacherForcing
                            NNModels{lay}.TeacherForcedTarget = Y_target;
                        else
                            
                        end
                        
                        if classificationproblem
                            Y_target = cat( 2 , Y_target(:,2:end,:) , Padding );
                        end
                        
                    case 'Attention' % LSTM to AttnNet to LSTM - UNDEVELOPED
                    case 'TimeDistributed' % LSTM to Dense
                        Xin_k = NNModels{lay-1}.Activations.HiddenOut;
                        TDistOut{lay} = zeros( [size(NNModels{lay}.Activations.HiddenOut) , size(Xin_k,2)]);
                    case 'regular' % LSTM to LSTM  ||  Dense to Dense || LSTM(last t) to Dense || Dense to LSTM (Sequence Embedding)
                        if contains(NNModels{lay-1}.Type,["LSTM","BiLSTM"]) && NNModels{lay}.Type=="dense"
                            Xin_k = permute( NNModels{lay-1}.Activations.HiddenOut(:,end,:) ,[1 3 2] ); % Last LSTM state into Dense
                        elseif NNModels{lay-1}.Type=="dense" && contains(NNModels{lay}.Type,["LSTM","BiLSTM"]) && LayerOperation(1,lay-1)=="SequenceEmbedding"
                            Xin_k = permute( SeqEmbOut{lay-1} , [1 3 2] );
                        else
                            Xin_k = NNModels{lay-1}.Activations.HiddenOut;
                        end
                end
            end
            
            switch NNModels{lay}.Type
                case 'dense' % PROPGATE DENSE
                    
                    switch LayerOperation(1,lay) % Activate Layer based on Operator Method
                        case 'TimeDistributed' % LSTM to Dense
                            for ti = 1:NNModels{lay-1}.Nstates
                                Xin_tk = permute( Xin_k(:,ti,:) ,[1 3 2] ); % reshape for fully connected dense layer
                                [NNModels{lay}] = NNPropogate( NNModels{lay} , Xin_tk , 'forward' , fcn_types );
                                TDistOut{lay}(:,:,ti) = NNModels{lay}.Activations.HiddenOut;
                                if lay==Nlayers
                                    OutActiv(:,ti,:) = permute( TDistOut{lay}(:,:,ti) , [1 3 2] );
                                end
                            end
                            if ~all(size(OutActiv) == size(Y_target))
                                disp('WRONG RESIZING'), beep , keyboard , end
                            
                        case 'SequenceEmbedding' % 'dense' to 'LSTM'
                            
                            if UseInputMasking
                                [~ ,cl] = find( ~InputMask{btc} );
                                lastStep = max(cl);
                            else
                                lastStep = size(Xin_k,2);%NNModels{lay+1}.Nstates;
                            end
                            
                            Xin_kEmd = Xin_k;
                            for ti = 1 : lastStep
                                Xin_tk = permute( Xin_kEmd(:,ti,:) ,[1 3 2] );
                                [NNModels{lay}] = NNPropogate( NNModels{lay} , Xin_tk , 'forward' , fcn_types );
                                SeqEmbOut{lay}(:,:,ti) = permute( NNModels{lay}.Activations.HiddenOut , [1 3 2] );
                            end
                            
                            if lay==Nlayers
                                OutActiv(:,:,:) = permute( SeqEmbOut{lay} , [1 3 2] );
                            end
                            
                        case 'regular' % Dense to Dense || LSTM(last t) to Dense
                            NNModels{lay} = NNPropogate( NNModels{lay} , Xin_k , 'forward' , fcn_types );
                            if lay==Nlayers
                                OutActiv = permute( NNModels{lay}.Activations.HiddenOut ,[1 3 2] );
                            end
                    end
                    
                    StoreLayerInputs{lay} = Xin_k;
                    
                case 'LSTM' % PROPGATE LSTM
                    
                    if UseInputMasking
                        if NNModels{lay}.InputMask.hasMask
                            NNModels{lay}.InputMask.Mask = InputMask{btc};
                        end
                    end
                    
                    switch LayerOperation(1,lay) % Activate Layer based on Operator Method
                        
                        case 'RepeatVector' % only for 2nd or deeper layers in a network
                            if NNModels{lay}.resetstate
                            else % Seed states with prior layers ending state
                                if seedStateFromPriorLayer==true
                                    NNModels{lay}.Inistates.Hidden = NNModels{lay-1}.Activations.HiddenOut(:,end,:);
                                    NNModels{lay}.Inistates.Cell = NNModels{lay-1}.Activations.Memory(:,end,:);
                                elseif btc>1 % seed cell state from PRIOR BATCH, SAME LAYER
                                    %%% TRIALING SETTING NEW STATE TO THE FINAL IME STEP, INSTEAD OF THE CORRESPONDING ELEMENT IN THE SEQUENCE
                                    NNModels{lay}.Inistates.Hidden = NNModels{lay}.Activations.HiddenOut(:, end ,:);
                                    NNModels{lay}.Inistates.Cell = NNModels{lay}.Activations.Memory(:,  end  ,:);
                                else % reset state, as its the first batch in theis epoch
                                    NNModels{lay}.Inistates.Hidden = zeros( size(NNModels{lay}.Inistates.Hidden) );
                                    NNModels{lay}.Inistates.Cell = zeros( size(NNModels{lay}.Inistates.Hidden) );
                                end
                            end
                            [NNModels{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'forward' , fcn_types , NNModels{lay}.Inistates );
                            
                            StoreLayerInputs{lay} = Xin_k;
                            
                            if lay==Nlayers
                                if NNModels{lay}.predictsequence
                                    OutActiv = NNModels{lay}.Activations.HiddenOut;
                                else
                                    OutActiv = permute( NNModels{lay}.Activations.HiddenOut(:,end,:) , [3 2 1] );
                                end
                            end
                            
                        case 'SelfReferencing' % only for 2nd or deeper layers in a network
                            
                            if NNModels{lay}.Attention % if attention mechanism, Encoder must have been in the layer prior.
                                NNModels{lay}.AttnInfo.EncoderInput = NNModels{ lay-1 }.Activations.HiddenOut(: , 1:NNModels{ lay-1 }.curStep , : );
                                NNModels{lay}.AttnInfo.AttentionScores = zeros( size(NNModels{lay}.AttnInfo.AttentionScores));
                            end
                            
                            % Seed states with prior layers ending state
                            endseq = NNModels{lay-1}.curStep;
                            NNModels{lay}.Inistates.Hidden = NNModels{lay-1}.Activations.HiddenOut(:,endseq,:);
                            NNModels{lay}.Inistates.Cell = NNModels{lay-1}.Activations.Memory(:,endseq,:);

                                %%% EITHER PROJECT & EMBED IN EACH TIMESTEP OPUTSIDE THE PROPOGHATE FUNCTION, OR INSIDE IT.
                                %%% IF INSIDE: "Xin_K" IS THE INPUT TO THE EMBEDDING, NOT THE LSTM LAYER, INCONSISTENT WITH OTHER USES OF "Xin_K".
                                [NNModels{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'forward' , fcn_types , NNModels{lay}.Inistates );
                                StoreLayerInputs{lay} = NNModels{lay}.XInput; % stored the inputs to the LSTM cell, AFTER projection layer, embedding layer, and teacher forcing
                            
                            
                            if lay==Nlayers
                                if NNModels{lay}.predictsequence
                                    OutActiv = NNModels{lay}.ProjectionLayerOutput;
                                else
                                    OutActiv = permute( NNModels{lay}.ProjectionLayerOutput(:,end,:) , [3 2 1] );
                                end
                            end
                            
                        case 'Attention'
                        case 'regular' % LSTM (all t) to LSTM (all t)
                            if NNModels{lay}.resetstate
                                % Leave initialised states for this layer unchanged
                            else % Initialise the State for Encoder starting position
                                if btc>1
                                    Ini_LastBatch{1,lay}.Hidden = NNModels{lay}.Activations.HiddenOut;
                                    Ini_LastBatch{1,lay}.Cell = NNModels{lay}.Activations.Memory;
                                elseif btc==1 % reset the state for the first batch in the training epoch
                                    Ini_LastBatch{1,lay}.Hidden = zeros( size(NNModels{lay}.Activations.HiddenOut) );
                                    Ini_LastBatch{1,lay}.Cell = zeros( size(NNModels{lay}.Activations.Memory) );
                                end
                                [NNModels{lay}.Inistates] = InitialiseLSTMNeuron( NNModels{lay}.Nstates, batchsize, Ini_PriorEpochThisbatch{btc,lay}, Ini_LastBatch{1,lay});
                                
                                %%% TRIALING SETTING NEW STATE TO THE FINAL IME STEP, INSTEAD OF THE CORRESPONDING ELEMENT IN THE SEQUENCE
                                if btc>1
                                    NNModels{lay}.Inistates.Hidden = NNModels{lay}.Activations.HiddenOut(:,end,:);
                                    NNModels{lay}.Inistates.Cell = NNModels{lay}.Activations.Memory(:,end,:);
                                else %% START OF NEW EPOCH = RESET STATES
                                    NNModels{lay}.Inistates.Hidden = zeros( size(NNModels{lay}.Inistates.Hidden(:,end,:) ) );
                                    NNModels{lay}.Inistates.Cell = zeros( size(NNModels{lay}.Inistates.Cell(:,end,:) ) );
                                end
                            end
                            
                            [NNModels{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'forward' , fcn_types , NNModels{lay}.Inistates );
                            if ~NNModels{lay}.resetstate
                                % store States for next epoch initialisation
                                Ini_PriorEpochThisbatch{btc,lay}.Hidden = NNModels{lay}.Activations.HiddenOut;
                                Ini_PriorEpochThisbatch{btc,lay}.Cell = NNModels{lay}.Activations.Memory;
                                
                            end
                            
                            if lay==Nlayers
                                if NNModels{lay}.predictsequence
                                    OutActiv = NNModels{lay}.Activations.HiddenOut;
                                else
                                    OutActiv = permute( NNModels{lay}.Activations.HiddenOut(:,end,:) , [3 2 1] );
                                end
                            end
                            
                            StoreLayerInputs{lay} = Xin_k;
                    end
                    
                    
            end
            
            
        end
        
        if UseOutputerrorMasking
            dCda = Loss_delta( OutActiv , Y_target ).*~OutputMask{btc};
        else
            dCda = Loss_delta( OutActiv , Y_target );
        end
        if gradnormalisation==true
            dCda_norm = gradnorm( permute( dCda(:,:,:),[3 2 1]) ); % gradient vector norms, for each timestep & feature
            dCda_norm_avg(btc) = mean( mean( dCda_norm , 3 ) ); %average over each batch, for each timestep
        end
        if any(isnan(dCda(:)))
            keyboard
        end
        
        % Loss/Cost of this prediction
        if classificationproblem
            if UseOutputerrorMasking
                ErrorMetric1( btc , it, : ) = mean(LossFcn( permute(OutActiv(:,:,:).*~OutputMask{btc},[1 3 2]) , ( permute(Y_target(:,:,:).*~OutputMask{btc},[1 3 2]) ) ),1);
                % sum( mean(LossFcn( (OutActiv(:,:,:)) , ( Y_target(:,:,:)) ),1) , 3);%  mean( arrayfun(@(btsz) mean(LossFcn( squeeze(OutActiv(btsz,:,:)) , squeeze( Y_target(btsz,:,:)) )), [1:batchsize]' ));
                % all timestpes in the observation must euqal ground truth, for the prediction to be correct
                ErrorMetric2( btc , it, : ) = mean( all(one_hot_convert(OutActiv(:,:,:),'decode',size(Y_target,3),  1  ,size(Y_target,3)).*~OutputMask{btc}...
                    ==one_hot_convert(Y_target(:,:,:),'decode',size(Y_target,3), 1 ,size(Y_target,3)).*~OutputMask{btc},2),1);
            else
                ErrorMetric1( btc , it, : ) = mean(LossFcn( permute(OutActiv(:,:,:) ,[1 3 2]) , ( permute(Y_target(:,:,:) ,[1 3 2]) ) ),1);% sum( mean(LossFcn( (OutActiv(:,:,:)) , ( Y_target(:,:,:)) ),1) , 3);%  mean( arrayfun(@(btsz) mean(LossFcn( squeeze(OutActiv(btsz,:,:)) , squeeze( Y_target(btsz,:,:)) )), [1:batchsize]' ));
                % all timestpes in the observation must euqal ground truth, for the prediction to be correct
                if size(Y_target,3)>1
                ErrorMetric2( btc , it, : ) = mean( all(one_hot_convert(OutActiv(:,:,:),'decode',size(Y_target,3),  1  ,size(Y_target,3))....
                    ==one_hot_convert(Y_target(:,:,:),'decode',size(Y_target,3), 1 ,size(Y_target,3)),2),1);
                else
                    ErrorMetric2( btc , it, : ) = mean(all(round(OutActiv)==Y_target,2));
                end
            end
            %mean(one_hot_convert(OutActiv(:,:,:),'decode',size(Y_target,3),  1  ,size(Y_target,3))==one_hot_convert(Y_target(:,:,:),'decode',size(Y_target,3), 1 ,size(Y_target,3)),1); % mean(arrayfun(@(btsz) mean(arrayfun(@(tt) one_hot_convert(OutActiv(btsz,tt,:),'decode',size(Y_target,3),0,size(Y_target,3))==one_hot_convert(Y_target(btsz,tt,:),'decode',size(Y_target,3),0,size(Y_target,3)),[1:size(Y_target,2)] ,'un',1)) ,[1:batchsize]' ));
        else
            ErrorMetric1( btc , it, : ) = mean( arrayfun(@(btsz) LossFcn( OutActiv(btsz,:,:) ,  Y_target(btsz,:,:) ) ,[1:batchsize]') ); %Y(:,fliplr(Prd_ti),:) );
        end
        
        Prediction( sampleidx(:,btc) , : , :) = OutActiv;
        
        if predictonly==true
            lay=lt(end);
            SavedStates{lay}.Inistates.Cell( sampleidx(:,btc) , 1 , : ) = NNModels{lay}.Inistates.Cell;
            SavedStates{lay}.Inistates.Hidden( sampleidx(:,btc) , 1 , : ) = NNModels{lay}.Inistates.Hidden;

            if NNModels{lay}.Attention
                AttnScores{btc,1} = NNModels{lay}.AttnInfo.AttentionScores;
            end
            
            if btc==size(DataX_batched,1)
                t2p = toc;
                if displaytrainingprogress
                sprintf('Prediction Complete. \n ErrorMetric 1: %0.2f \n ErrorMetric 2: %0.2f \n time to Prediction: %0.0f seconds'...
                    ,mean( mean(ErrorMetric1(:,it,:),1),3,'omitnan') ...
                    ,mean( mean(ErrorMetric2(:,it,:),1),3,'omitnan') ...
                    ,t2p )
                end
                
                if nargout>4
                    varargout{1} = SavedStates;
                    if NNModels{lay}.Attention
                    AttnScores = cat(1,AttnScores{:});
                    varargout{2} = AttnScores;
                    else
                        varargout{2} =[];
                    end
                end
                return
            else
                
            end
        elseif predictonly==false
            
            
            %% BACKWARD PROPOGATE
            for lay = Nlayers:-1:1
                Xin_k = StoreLayerInputs{lay};
                
                if lay==Nlayers
                    delXBP = dCda;
                else
                    if contains(NNModels{lay}.Type,["LSTM","BiLSTM"]) && NNModels{lay+1}.Type=="dense"% Last LSTM state into Dense
                        if NNModels{lay}.predictsequence==false
                            delXBP = permute( delX{lay+1} , [1 3 2]);
                        else
                            delXBP = delX{lay+1};
                        end
                    else
                        if LayerOperation(1,lay+1)=="RepeatVector" % into this layer was the final LSTM output on the layer prior, repeated. So the error passed backwards needs to be only for the prioer layers final output, and should be the average of all the DelX erorrs found across all timestpes of this layer
%                             delXBP = [ zeros(size(delX{lay+1}(:,1:end-1,:)) ) , mean( delX{lay+1} , 2 ) ];
%                             delXBP = [ zeros(size(delX{lay+1}(:,1:end-1,:)) ) , sum( delX{lay+1} , 2 ) ];
                            delXBP = [ zeros( size(NNModels{lay}.Activations.HiddenOut(:,1:end-1,:)) ) , sum( delX{lay+1} , 2 ) ];
                        elseif LayerOperation(1,lay+1)=="SelfReferencing"
                            if NNModels{lay+1}.('Attention')
                                delXBP = NNModels{lay+1}.AttnInfo.Enc_del_total;
                            else
                                delXBP = zeros( size(NNModels{lay}.Activations.HiddenOut) );
                            end
                        else
                            delXBP = delX{lay+1};
                        end
                    end
                end
                
                switch NNModels{lay}.Type
                    case 'dense'
                        
                        switch LayerOperation(1,lay) % Activate Layer based on Operator Method
                            case 'TimeDistributed' % LSTM to Dense
                                for ti = 1:NNModels{lay-1}.Nstates
                                    Xin_tk = permute( Xin_k(:,ti,:) ,[1 3 2] ); % reshape for fully connected dense layer
                                    delXBP_t = permute( delXBP(:,ti,:) ,[1 3 2] );                      if lay~=Nlayers , beep , disp('should be the final layer') , keyboard , end
                                    [NNModels{lay}, delXf] = NNPropogate( NNModels{lay} , Xin_tk , 'backward' , fcn_types  , delXBP_t );
                                    delX{lay}(:,ti,:) = permute( delXf ,[1 3 2]);
                                    TimeDistributedErrorUpdates.wIN = TimeDistributedErrorUpdates.wIN + NNModels{lay}.dEdW.wIN;
                                    TimeDistributedErrorUpdates.wb = TimeDistributedErrorUpdates.wb + NNModels{lay}.dEdW.wb;
                                    if ti==NNModels{lay-1}.Nstates
                                        NNModels{lay}.dEdW = TimeDistributedErrorUpdates;% ./ NNModels{lay-1}.Nstates; % store the time-averaged Error updates for this Batch of data
                                        TimeDistributedErrorUpdates = TDEUini; % reset to an empty value
                                    end
                                end
                                
                            case 'regular' % Dense to Dense || LSTM(last t) to Dense
                                if lay>1
                                    if NNModels{lay-1}.Type=="LSTM"
                                        [NNModels{lay}, delX{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'backward' , fcn_types , permute( delXBP(:,end,:) ,[1 3 2] ) );
                                    else
                                        [NNModels{lay}, delX{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'backward' , fcn_types , delXBP );
                                    end
                                else
                                    [NNModels{lay}, delX{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'backward' , fcn_types , delXBP );
                                end
                                
                            case 'SequenceEmbedding'
                                
                                Emb_dEdW{lay}.dEdW = Emb_dEdW{lay}.InitilaiseValues;
                                
                                for ti = size(Xin_k,2):-1:1 % NNModels{lay+1}.Nstates:-1:1
                                    % BProp LSTM error to the embedding layer
                                    [NNModels{lay}, delX{lay}(:,ti,:)] = NNPropogate( NNModels{lay} , permute( Xin_k(:,ti,:) ,[1 3 2] ) , 'backward' , fcn_types , permute( delXBP(:,ti,:) ,[1 3 2] ) );
                                    Emb_dEdW{lay}.dEdW.wIN = Emb_dEdW{lay}.dEdW.wIN + NNModels{lay}.dEdW.wIN;
                                    Emb_dEdW{lay}.dEdW.wb = Emb_dEdW{lay}.dEdW.wb + NNModels{lay}.dEdW.wb;
                                    
                                end
                                
                                % Time average the Error updates for the Input data embedding layer
                                NNModels{lay}.dEdW.wIN = Emb_dEdW{lay}.dEdW.wIN;%./ NNModels{lay+1}.Nstates;
                                
                        end
                        
                        
                    case 'LSTM'
                        
                        switch LayerOperation(1,lay) % Activate Layer based on Operator Method
                            case 'SelfReferencing'
                                % this layer produces a context vector, to pass into the next self referencing layer (decoder)
                                % Pass back the error from the cell state of the LSTM layer in layer+1
                                
                                % BProp Projection Error to the LSTM
                                [NNModels{lay}, delX{lay}, delState{lay}, del_recc_BP{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'backward' , fcn_types , delXBP );
                                
                                % Backpropogated error through input to embedding layer attached to this LSTM does not get passed on to earlier layers
                                delX{lay}(:,:,:) = [0];
                                
                            case 'regular'
                                
                                if lay~=Nlayers
                                    delStateBP = delState{lay+1};
                                    delreccBP = del_recc_BP{lay+1};
                                    [NNModels{lay}, delX{lay}, delState{lay}, del_recc_BP{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'backward' , fcn_types , delXBP , delStateBP , delreccBP );
                                else
                                    [NNModels{lay}, delX{lay}, delState{lay}, del_recc_BP{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'backward' , fcn_types , delXBP );
                                end
                                
                            case 'RepeatVector'
                                
                                    [NNModels{lay}, delX{lay}, delState{lay}, del_recc_BP{lay}] = NNPropogate( NNModels{lay} , Xin_k , 'backward' , fcn_types , delXBP );
                                
                        
                        end
                        
                end
                
            end
            
            
            % Plot
            if plottingiterations==true
                if it==1 && btc==1
                    plthandles=[];
                end
                if btc==size(DataX_batched,1)
                    [plthandles] = PlotTraining( NNModels , (it-1)*batchsize + btc , plthandles , mean( mean(ErrorMetric1( btc , it, : ),1),3) , mean(dCda(:)) );
                    
                    if rem( it , floor(epochs/4))==0 || it==1
                        [StoreGrads] = TraceBackpropError(NNModels,dCda,X_input,fcn_types);
                        if ~exist('sbh','var') , sbh = subplot(2,2,4); end
                        bar(sbh,StoreGrads.BPseq);
                    end
                end
                
            end
            
            
            % Update The Weights
            if strcmpi(GDOptimizer,'Vanilla') || strcmpi(GDOptimizer,'Adam')
                OptParams{2} = it;
            end
            for lay = 1:Nlayers
                if NNModels{lay}.Type=="LSTM"
                    if NNModels{lay}.SelfReferencing
                        [NNModels{lay}.ProjectionLayer] = NNWeightUpdate( NNModels{lay}.ProjectionLayer , lrsched(it) , GDOptimizer , OptParams , gradclip );
                        [NNModels{lay}.ProjectionLayer] = NNClearWeightErrors( NNModels{lay}.ProjectionLayer );
                        if ~isempty( NNModels{lay}.EmbeddingLayer)
                            [NNModels{lay}.EmbeddingLayer] = NNWeightUpdate( NNModels{lay}.EmbeddingLayer , lrsched(it) , GDOptimizer , OptParams , gradclip );
                            [NNModels{lay}.EmbeddingLayer] = NNClearWeightErrors( NNModels{lay}.EmbeddingLayer );
                        end
                        if NNModels{lay}.Attention
                            if contains( "Weights", fieldnames(NNModels{lay}.AttnInfo) )
                                [NNModels{lay}.AttnInfo] = NNWeightUpdate( NNModels{lay}.AttnInfo , lrsched(it) , GDOptimizer , OptParams , gradclip );
                                [NNModels{lay}.AttnInfo] = NNClearWeightErrors( NNModels{lay}.AttnInfo );
                            end
                        end
                    end
                    if NNModels{lay}.BiLSTM
                            [NNModels{lay}.Forward] = NNWeightUpdate( NNModels{lay}.Forward , lrsched(it) , GDOptimizer , OptParams , gradclip );
                            [NNModels{lay}.Backward] = NNWeightUpdate( NNModels{lay}.Backward , lrsched(it) , GDOptimizer , OptParams , gradclip );
                    else
                        [NNModels{lay}] = NNWeightUpdate( NNModels{lay} , lrsched(it) , GDOptimizer , OptParams , gradclip );
                    end
                end
                    
                        
                
                if NNModels{lay}.Type=="dense"
                    [NNModels{lay}] = NNWeightUpdate( NNModels{lay} , lrsched(it) , GDOptimizer , OptParams , gradclip );
                    [NNModels{lay}] = NNClearWeightErrors( NNModels{lay} );
                end
            end
            
        end
    end
    
    if displaytrainingprogress && rem( it , floor(epochs/40))==0
        e1 = mean(mean( ErrorMetric1(:,it,:),1,'omitnan'),3);
        e2 = mean(mean( ErrorMetric2(:,it,:),1,'omitnan'),3); if isnan(e2) , e2=[]; end
        if floor(epochs/40)==it
            maxspc = quantile( reshape(mean( ErrorMetric1(:,1:it,:) ,3),1,it*size(ErrorMetric1,1)) ,0.50);% minspc = min( reshape(mean( ErrorMetric1(:,1:it,:) ,3),1,it*size(ErrorMetric1,1)) );
        end
        cgap = repmat(' ',1, ceil(70.*(e1/maxspc)) );
        fprintf('ErMet1(%0.0f/%0.0f): %s %0.2f|%0.0f%%   gnrm:%0.2f \n', it/(floor(epochs/40)) , epochs/ ( floor(epochs/40) ) ,cgap ,e1,round(e2*100,0), dCda_norm_avg(btc) ); drawnow;
        if it>ceil(epochs/10)
            if isnan(e2)
                ERVAL = mean(mean( ErrorMetric1(:,it-5:it,:),1,'omitnan'),3);
                ERVALCHK = ERVAL<0.001;
            else
                ERVAL = mean(mean( ErrorMetric2(:,it-5:it,:),1,'omitnan'),3);
                ERVALCHK = all(ERVAL>0.97);
            end
            if ERVALCHK
                fprintf(' CONVERGENCE - Acc. %0.0f',mean(mean( ErrorMetric2(:,it-5:it,:),1,'omitnan'),3))
                try
                      % add to total training iteration counter
                      NNModels{end}.trainingInformation = NetworkSettings;
                      
                    if contains( "NumEpochs", fieldnames(NNModels{end}) ) && predictonly==false
                        NNModels{end}.NumEpochs = NNModels{end}.NumEpochs + it;
                        
                        if contains( "LossHistory", fieldnames(NNModels{end}) )
                            NNModels{end}.LossHistory = [NNModels{end}.LossHistory, mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        else
                            NNModels{end}.LossHistory = [mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        end
                    elseif  predictonly==false
                        NNModels{end}.NumEpochs = it;
                        
                        if contains( "LossHistory", fieldnames(NNModels{end}) )
                            NNModels{end}.LossHistory = [NNModels{end}.LossHistory, mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        else
                            NNModels{end}.LossHistory = [mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        end
                        NNModels{end}.LossHistory = [mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                    end
                catch
                    keyboard
                end
                save('TrainingCheckpoint.mat')
                try
                      % plot PCA components in 2D of observations to see if they
                      % are diverse or not.
                      LayerStatePCAcheck
                catch
                end
                
                return %model converged, exit training
                
            end
        end
    end
    if displaytrainingprogress && rem( it , floor(epochs/20))==0
        if any(lt)
            
            %             % Display error hotspots
            %             for btsz=1:300
            %                 LossBlock(btsz,:) = LossFcn( squeeze(Prediction(btsz,:,:)) , squeeze( DataY_batched{btsz,1}) )';
            %             end
            %             figure; subplot(3,1,1) , imagesc( squeeze( mean( cat(1,DataY_batched{:,1}) ,1) ) ) , ...
            %                 subplot(3,1,2) , imagesc( squeeze( mean(Prediction(:,:,:),1) ) )
            %             subplot(3,1,3) , imagesc( squeeze( mean(LossBlock(:,:,:),1) ) )
            
            
            %             mn = arrayfun(@(btc) mean( Ini_PriorEpochThisbatch{btc,lt(1)}.Cell(:,1,:),3) , [1:Nbatches]' ,'un',0);
            %             sd = arrayfun(@(btc) std( Ini_PriorEpochThisbatch{btc,lt(1)}.Cell(:,1,:),0,3) , [1:Nbatches]' ,'un',0);
            %             clf; subplot(2,1,1); plot( [1:Nobservations]' ,[ [cat(1,mn{:})] , cell2mat(DataY_batched)] ); yyaxis right;
            %             plot( [1:Nobservations]' , [cat(1,sd{:})] );legend({'cell initial','Y target','cell Std dev'}); subplot(2,1,2); plot([Prediction, cell2mat(DataY_batched)]); drawnow;
        end
    end
    
end
% add to total training iteration counter
NNModels{end}.trainingInformation = NetworkSettings;


                    if contains( "NumEpochs", fieldnames(NNModels{end}) ) && predictonly==false
                        NNModels{end}.NumEpochs = NNModels{end}.NumEpochs + it;
                        if contains( "LossHistory", fieldnames(NNModels{end}) )
                            NNModels{end}.LossHistory = [NNModels{end}.LossHistory, mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        else
                            NNModels{end}.LossHistory = [mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        end
                    elseif  predictonly==false
                        NNModels{end}.NumEpochs = it;
                        if contains( "LossHistory", fieldnames(NNModels{end}) )
                            NNModels{end}.LossHistory = [NNModels{end}.LossHistory, mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        else
                            NNModels{end}.LossHistory = [mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                        end
                        NNModels{end}.LossHistory = [mean(mean(ErrorMetric1(:,1:it,:),1),3)];
                    end
                    
                
t2f = toc;
if displaytrainingprogress
sprintf('Training Complete. \n ErrorMetric 1: %0.2f \n ErrorMetric 2: %0.2f \n time to train: %0.0f seconds'...
    ,mean( mean(ErrorMetric1(:,end,:),1),3,'omitnan') ...
    ,mean( mean(ErrorMetric2(:,end,:),1),3,'omitnan') ...
    ,t2f )
disp(t2f)
end
if ~predictonly
save('TrainingCheckpoint.mat')
end
end