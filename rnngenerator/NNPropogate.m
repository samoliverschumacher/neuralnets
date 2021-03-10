function [NNLayer , varargout] = NNPropogate( NNLayer , DataIN , propdirection , fcn_types , varargin )

switch NNLayer.Type
    
    case 'LSTM'
        
        % flag to process each observation in sequence rather than in parralel, so time sequence initialisiation uses prior observations 1st output as its initial value
        if NNLayer.PropBatchinSequence==true
            batchiterate = 1:size(DataIN,1);
        else
            batchiterate = [ 1:size(DataIN,1) ]';
        end
        
        if NNLayer.BiLSTM
            switch propdirection
                case 'forward'
                    
                    Xin_forward = DataIN;
                    Xin_backward = fliplr(DataIN);
                    
                    [NNLayer.Forward] = NNPropogate( NNLayer.Forward , Xin_forward , 'forward' , fcn_types );
                    [NNLayer.Backward] = NNPropogate( NNLayer.Backward , Xin_backward , 'forward' , fcn_types );
                    
                    NNLayer.curStep = NNLayer.Forward.curStep;
                    
                    switch NNLayer.MergeFcn
                        case "concatenate" % Concatenate output
                            NNLayer.BiCellOut = cat( 3 , NNLayer.Forward.Activations.Memory , NNLayer.Backward.Activations.Memory );
                            NNLayer.BiHout = cat( 3 , NNLayer.Forward.Activations.HiddenOut , NNLayer.Backward.Activations.HiddenOut );
                        case "sum" % sum froward and backward vectors. Scoring function cannot be dotproduct as size(Encoder)~=size(Decoder)
                            NNLayer.BiCellOut = sum( cat( 3 , NNLayer.Forward.Activations.Memory , NNLayer.Backward.Activations.Memory ) , 3);
                            NNLayer.BiHout = sum( cat( 3 , NNLayer.Forward.Activations.HiddenOut , NNLayer.Backward.Activations.HiddenOut ) , 3);
                    end
                    
                    NNLayer.Activations.Memory = NNLayer.BiCellOut;
                    NNLayer.Activations.HiddenOut = NNLayer.BiHout;
                    
                case 'backward'
                    
                    deltaErr = varargin{1};
                    delStateBP = varargin{2};
                    delreccBP = varargin{3};
                    
                    switch NNLayer.MergeFcn
                        case "concatenate" % Concatenate output
                            deltaErr_forward = deltaErr(:,:,1:NNLayer.Forward.Nunits);
                            deltaErr_backward = deltaErr(:,:, NNLayer.Forward.Nunits+1:end);
                            
                            delStateBP_forward = delStateBP(:,:,1:NNLayer.Forward.Nunits);
                            delStateBP_backward = delStateBP(:,:,NNLayer.Forward.Nunits+1:end);
                            
                            delreccBP_forward = delreccBP(:,:,1:NNLayer.Forward.Nunits);
                            delreccBP_backward = delreccBP(:,:,NNLayer.Forward.Nunits+1:end);
                        case "sum"
                            [deltaErr_forward , deltaErr_backward] = deal(deltaErr);
                            [delStateBP_forward , delStateBP_backward] = deal(delStateBP);
                            [delreccBP_forward , delreccBP_backward] = deal(delreccBP);
                    end
                    
                    Xin_forward = DataIN;
                    Xin_backward = fliplr(DataIN);
                    
                    [NNLayer.Forward, delXFwd ] = NNPropogate( NNLayer.Forward , Xin_forward , 'backward' , fcn_types , deltaErr_forward , delStateBP_forward , delreccBP_forward);
                    [NNLayer.Backward, delXBwd ] = NNPropogate( NNLayer.Backward , Xin_backward , 'backward' , fcn_types , deltaErr_backward , delStateBP_backward , delreccBP_backward);
                    
                    % Error propogated back to Embedding (or prior layer)
                    %                 d_x = sum( delXFwd , delXBwd , 3);
                    d_x = mean( cat(4,delXFwd , delXBwd) , 4 ); % to make the erors not as dramatic.
                    
                    varargout{1} = d_x;
                    varargout{2} = [];%d_x(:,1,:);
                    varargout{3} = [];%d_x(:,1,:);
            end
            
        else % Regular LSTM
            
            switch propdirection
                
                case 'forward'
                    if numel(varargin)==1
                        H_ini = varargin{1}.Hidden;
                        C_ini = varargin{1}.Cell;
                    else
                        [H_ini,C_ini] = deal( zeros( size(NNLayer.Activations.HiddenOut(:, 1 ,:) )) );
                    end
                    
                    if NNLayer.InputMask.hasMask
                        [~ ,cl] = find(~NNLayer.InputMask.Mask);
                        lastStep = max(cl);
                        firstStep = min(cl);
                        statesequence = 1:lastStep;
                    else
%                         statesequence = 1:NNLayer.Nstates;
                        statesequence = 1:size(DataIN,2);
                    end
                    
                    
                    for smpl = batchiterate
                        for tt = statesequence
                            % keep track of current step in sequence
                            NNLayer.curStep = tt;
                            
                            if tt==1 % initialise with prior layers Memory and Hidden States
                                %%% TRIALLING INITILAISING STATES FROM PRIOR OBSERVATION IN A BATCH
                                if NNLayer.PropBatchinSequence==true && all(smpl>1) % get state from the prior sample in this batch, in the same equivalent timestep
                                    Hin = cat( 1 , ( permute( NNLayer.Activations.HiddenOut( smpl-1 , lastStep ,:) ,[1 3 2]) ) );
                                    memcellIn = cat( 1 , ( permute( NNLayer.Activations.Memory( smpl-1 , lastStep ,:) ,[1 3 2]) ) );
                                else
                                    Hin = cat( 1 , ( permute( H_ini(smpl,:,:) ,[1 3 2]) ) ); % cat( 1 , ( permute( zeros( size(NNLayer.Activations.HiddenOut(:, 1 ,:)) ) ,[1 3 2]) ) ); % [Samples x Timesteps x Features] --> [Samples x Features] %                         Hin = zeros(NNLayer.Activations.HiddenOut(:, 1 ,:)); %[ 0 ];
                                    memcellIn = cat( 1 , ( permute( C_ini(smpl,:,:) ,[1 3 2]) ) ); % cat( 1 , ( permute( zeros( size(NNLayer.Activations.Memory(:, 1 ,:)) ) ,[1 3 2]) ) );
                                end
                            else % from prior timestep
                                Hin = cat( 1 , ( permute( NNLayer.Activations.HiddenOut(smpl,tt - 1,:) ,[1 3 2]) ) ); % [Samples x Timesteps x Features] --> [Samples x Features] %                         Hin = NNLayer.Activations.HiddenOut(:,tt - 1,:);
                                memcellIn = cat( 1 , ( permute( NNLayer.Activations.Memory(smpl,tt - 1,:) ,[1 3 2]) ) );
                            end
                            NNLayer.Activations.HiddenIn(smpl,tt,:) = permute( Hin , [1 3 2] );
                            if tt==1
                                NNLayer.Activations.MemoryIn(smpl,tt,:) = permute( memcellIn , [1 3 2] );
                            end
                            
                            if NNLayer.SelfReferencing
                                
                                % Could make teacher forced tareget fields obsolete?
                                if tt==1
                                    TargetVect = cat( 1 , ( permute( DataIN(smpl,tt,:) ,[1 3 2]) ) );
                                else
                                    % Input to layer is last time steps hidden output.
                                    if (NNLayer.TeacherForcing) && (NNLayer.TeacherForcing)*rand() < NNLayer.teacherforcingratio % convert logical to 1|0 ,
                                        % randomly feed the correct(target) prior output, as an input into the current timestep
                                        
                                        TargetVect = cat( 1 , ( permute( NNLayer.TeacherForcedTarget(smpl,tt,:) ,[1 3 2]) ) );
                                    else % no teacher forcing, use the prior decoder output, as this input.
                                        % Project Hidden output to a Target Vector by passing it through a Fully Connected layer.
                                        
                                        TargetVect = permute( NNLayer.ProjectionLayerOutput(smpl,tt-1,:) , [1 3 2] );
                                    end
                                end
                                
                                if ~isempty(NNLayer.EmbeddingLayer)
                                    % Then, pass the token Expectation vector from the fully connected
                                    % layer into the embedding layer, so it can feed into the next timestep of the Decoder LSTM
                                    [NNLayer.EmbeddingLayer] = NNPropogate( NNLayer.EmbeddingLayer , TargetVect , 'forward', fcn_types );
                                    NNLayer.EmbeddingLayerInput(:,tt,:) = permute( TargetVect , [1 3 2] );
                                    Xin_t =  NNLayer.EmbeddingLayer.Activations.HiddenOut;
                                    % must have an attached embedding layer, so each timesteps output can be converted
                                else
                                    Xin_t = TargetVect;
                                end
                                
                                if NNLayer.Attention == true % https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3#ba24 % https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f
                                    if tt==1
                                        % Initialise the Decoder with the start token to get a Hidden state output so
                                        % that the Attention mechanism can create a context vecvtor for Input 2 (the first tiermstep to predict)
                                        Xin_t = cat( 2 , Xin_t ...
                                            , permute( H_ini(:,:,:) ,[1 3 2]) );
                                    else
                                        [ContextV, AttnScore] = AttentionLayerPropagate( NNLayer , Hin, 'forward' );
                                        NNLayer.AttnInfo.AttentionScores(:,1:size(NNLayer.AttnInfo.EncoderInput,2),tt) = AttnScore; %rows: input, cols: encoder timesteps, 3rd dim: output sequence timestep
                                        
                                        Xin_t = cat( 2 , Xin_t ...
                                            , ContextV );
                                        
                                    end
                                end
                                
                            else
                                
                                %                             if (NNLayer.TeacherForcing)*rand() > 0.5 % convert logical to 1|0 ,
                                % randomly feed the correct(target) prior output, as an input into the current timestep
                                %                                 Xin_t = NNLayer.TeacherForcedTarget(:,tt,:); %%%% IN DEVELOPMENT ---- INSERT THE PRIOR TARGET OUTPUT HERE
                                %                             else % no teacher forcing
                                Xin_t = cat( 1 , ( permute( DataIN(smpl,tt,:) ,[1 3 2]) ) );
                                %                             end
                                
                            end
                            NNLayer.XInput(smpl,tt,:) = permute( Xin_t , [1 3 2] );
                            
                            %% Sever Synapse!!
                            if NNLayer.Type=="LSTM" && ~NNLayer.SelfReferencing && tt==4 && (size(NNLayer.XInput,2)~=12)
                                %                             memcellIn = memcellIn.*0;
                                %                             Hin = Hin.*0;
                            end
                            
                            [NNLayer] = LSTMCell( memcellIn , Xin_t , Hin , NNLayer.Weights , fcn_types , NNLayer , tt , smpl );
                            %                         NNLayer.Activations.Memory(smpl,tt,:) = memcellOut;
                            %                         NNLayer.Activations.HiddenOut(smpl,tt,:) = hiddenOut;
                            %                         NNLayer.Activations.Gates.forget(smpl,tt,:) = gates.forget;
                            %                         NNLayer.Activations.Gates.input(smpl,tt,:) = gates.input;
                            %                         NNLayer.Activations.Gates.output(smpl,tt,:) = gates.output;
                            %                         NNLayer.Activations.Gates.activate(smpl,tt,:) = gates.activate;
                            
                            if NNLayer.SelfReferencing
                                
                                [NNLayer.ProjectionLayer] = NNPropogate( NNLayer.ProjectionLayer , permute( NNLayer.Activations.HiddenOut(smpl,tt,:) ,[1 3 2]) , 'forward', fcn_types );
                                NNLayer.ProjectionLayerOutput(:,tt,:) = permute( NNLayer.ProjectionLayer.Activations.HiddenOut , [1 3 2] );
                                
                            end
                            
                        end
                    end
                    
                    
                case 'backward'
                    
                    deltaErr = varargin{1}; % Error between this layers outputs, and the deeper layers target value. or this layers output and the final target output
                    
                    
                    % initialise
                    btSz = size(NNLayer.Activations.HiddenOut,1);
                    %                 sz_ijb = [size(DataIN,3) , NNLayer.Nunits, btSz];
                    %                 sz_iib = [NNLayer.Nunits , NNLayer.Nunits, btSz];
                    %                 sz_1ib = [ 1 , NNLayer.Nunits, btSz];
                    [NNLayer] = NNClearWeightErrors( NNLayer );
                    %                 dEdW = struct('wIN', struct('forget',zeros(sz_ijb),'input',zeros(sz_ijb) ,'activate',zeros(sz_ijb) ,'output',zeros(sz_ijb) ) ...
                    %                     ,'wrec', struct('forget',zeros(sz_iib) ,'input',zeros(sz_iib) ,'activate',zeros(sz_iib) ,'output',zeros(sz_iib) ) ...
                    %                     ,'wb', struct('forget',zeros(sz_1ib) ,'input',zeros(sz_1ib) ,'activate',zeros(sz_1ib) ,'output',zeros(sz_1ib) ) ...
                    %                     ,'wpeep', struct('forget',zeros(sz_iib) ,'input',zeros(sz_iib) ,'output',zeros(sz_iib) ) );
                    
                    sz_bsu = zeros( btSz , NNLayer.Nstates , NNLayer.Nunits );
                    
                    [d_x] = deal( zeros( btSz , NNLayer.Nstates , size(DataIN,3) ) );
                    [d_state, d_f, d_i, d_a, d_o] = deal( sz_bsu );
                    
                    [delout] = deal( sz_bsu ); % prior timesteps Hidden unit backpropogated error.
                    
                    [delErr , d_out] = deal( zeros( btSz , 1 , NNLayer.Nunits ) ); % doesnt need to have all timesteps stored
                    
                    % In the case of masked inputs; Last step the forward propogation got to for this batch
                    lastStep = NNLayer.curStep;
                    
                    if numel(varargin)>1
                        delState_bp = varargin{2}; % if this sequence ends by passing the Memory Cell onto another reccurent sequence

                        if numel(varargin)>2
                            delreccBP = varargin{3};
                            delout(:, lastStep ,:) = delreccBP; % if this sequence ends by passing the Hidden State onto another reccurent sequence
                        end
                    end
                    
                    
                    if NNLayer.SelfReferencing
                        if NNLayer.Attention
                            NNLayer.AttnInfo.Enc_del_total = zeros( size(NNLayer.AttnInfo.Enc_del_total) ); % initialise the storage of the back propagated error which will go through the encoder hidden output
                        end
                        %                     statesequence = NNLayer.Nstates:-1:1; % statesequence = NNLayer.curStep;
                        if ~isempty(NNLayer.EmbeddingLayer)
                            Emb_dEdW.wIN = zeros( size(NNLayer.EmbeddingLayer.dEdW.wIN) );
                            Emb_dEdW.wb = zeros( size(NNLayer.EmbeddingLayer.dEdW.wb) );
                        end
                        Prj_dEdW.wIN = zeros( size(NNLayer.ProjectionLayer.dEdW.wIN) );
                        Prj_dEdW.wb = zeros( size(NNLayer.ProjectionLayer.dEdW.wb) );
                        % Backpropogaed Error into this function is the error
                        % after Projection through the fully connected layer after the LSTM Hidden Output
                        deltaErr_Prj = varargin{1};
                        % Backpropogated Error to the LSTM will be defined by
                        % backpropogation through the projection layer first
                        deltaErr = zeros( size(NNLayer.Activations.HiddenOut) );
                    end
                    
                    statesequence = lastStep:-1:1;
                    
                    for tt = statesequence
                        % keep track of current step in sequence
                        NNLayer.curStep = tt;
                        
                        if NNLayer.SelfReferencing
                            
                            % Backpropogate the Prediction Error to the projection layer
                            XinPrj_t = permute( NNLayer.Activations.HiddenOut(:,tt,:) , [1 3 2] );% [Samples x Timesteps x Features] --> [Samples x Features]
                            
                            [NNLayer.ProjectionLayer, delXPrj] = NNPropogate( NNLayer.ProjectionLayer , XinPrj_t , 'backward' , fcn_types , permute( deltaErr_Prj(:,tt,:) , [1 3 2] ) );
                            Prj_dEdW.wIN = Prj_dEdW.wIN + NNLayer.ProjectionLayer.dEdW.wIN;
                            Prj_dEdW.wb = Prj_dEdW.wb + NNLayer.ProjectionLayer.dEdW.wb;
                            
                            deltaErr(:,tt,:) = permute( delXPrj , [1 3 2] );
                            Xin_t = permute( NNLayer.XInput(:,tt,:) , [1 3 2] );
                        else
                            Xin_t = cat( 1 , ( permute( DataIN(:,tt,:) ,[1 3 2]) ) ); % [Samples x Timesteps x Features] --> [Samples x Features]
                        end
                        
                        
                        
                        if tt == lastStep%NNLayer.Nstates% final timestep. prior Hidden Output backpropogated error (or prediction error).
                            delErr(:,1,:) = deltaErr(:, lastStep, :); % only final timestep has a backpropgated error
                        elseif NNLayer.predictsequence
                            delErr(:,1,:) = deltaErr(:, tt, :);
                        else
                            delErr(:,1,:) = zeros(size( deltaErr(:, end, :) )); % only final timestep has a backpropgated error
                        end
                        
                        if NNLayer.Attention
                            if tt < statesequence(end)
                                %                             delErr(:,1,:) = delErr(:,1,:) + Ht_prior_del; % attention layer concatenates with the prior Decoder output, to form the Decoder input vector. Backproagate the error from the decoder input, to the prior decoder output.
                            end
                        end
                        state_t = permute( NNLayer.Activations.Memory(:,tt,:) ,[1 3 2]);
                        
                        d_out(:, 1 ,:) = delErr(:,1,:) + delout(:,tt,:);  % Error passed from previous (future) reccurnt unit + Error of This units output to target.
                        if tt== lastStep%NNLayer.Nstates % There is no future cell state to backpropogate to this time step.
                            d_state(:,tt,:) = d_out(:, 1 ,:) .* NNLayer.Activations.Gates.output(:,tt,:) ...
                                .* permute( Actvfcn( state_t , true , fcn_types.(NNLayer.ActFcn) ), [1 3 2] );
                            if exist('delState_bp','var')
                                d_state(:,tt,:) = d_state(:,tt,:) + delState_bp;
                            end
                            
                        else
                            d_state(:,tt,:) = d_out(:, 1 ,:) .* NNLayer.Activations.Gates.output(:,tt,:) ...
                                .* permute( Actvfcn( state_t , true , fcn_types.(NNLayer.ActFcn) ), [1 3 2] ) ...
                                + d_state(:,tt+1,:).*NNLayer.Activations.Gates.forget(:,tt+1,:);
                            
                        end
                        
                        if tt==1 % There is no prior cell state, set cell state term to zero.
                            d_f(:,tt,:) =  d_state(:,tt,:) .* NNLayer.Activations.MemoryIn .* permute( Actvfcn( permute( NNLayer.Activations.GatesIn.forget(:,tt,:) , [1 3 2] )  , true , fcn_types.sigmoid ), [1 3 2] );%zeros( size( d_state(:,tt,:) ) ); % % d_state(:,tt,:) .* NNLayer.Activations.MemoryIn .* permute( Actvfcn( (Xin_t * NNLayer.Weights.wIN.forget ) + (Hin * NNLayer.Weights.wrec.forget)  + ( permute( zeros(size(NNLayer.Activations.Memory(:,tt,:))) ,[1 3 2] ) * NNLayer.Weights.wpeep.forget).*YP + NNLayer.Weights.wb.forget  , true , fcn_types.sigmoid ), [1 3 2] );
                            d_i(:,tt,:) = d_state(:,tt,:) .* NNLayer.Activations.Gates.activate(:,tt,:) .* permute( Actvfcn( permute( NNLayer.Activations.GatesIn.input(:,tt,:) , [1 3 2] ) , true , fcn_types.sigmoid ), [1 3 2] );
                        else
                            d_f(:,tt,:) = d_state(:,tt,:) .* NNLayer.Activations.Memory(:,tt-1,:) .* permute( Actvfcn( permute( NNLayer.Activations.GatesIn.forget(:,tt,:) , [1 3 2] ) , true , fcn_types.sigmoid ), [1 3 2] );
                            d_i(:,tt,:) = d_state(:,tt,:) .* NNLayer.Activations.Gates.activate(:,tt,:) .* permute( Actvfcn( permute( NNLayer.Activations.GatesIn.input(:,tt,:) , [1 3 2] ) , true , fcn_types.sigmoid ), [1 3 2] );
                        end
                        d_a(:,tt,:) = d_state(:,tt,:) .* NNLayer.Activations.Gates.input(:,tt,:) .* permute( Actvfcn( permute( NNLayer.Activations.GatesIn.activate(:,tt,:) , [1 3 2] )  , true , fcn_types.(NNLayer.ActFcn) ), [1 3 2] );
                        d_o(:,tt,:) = d_out(:, 1 ,:) .* Actvfcn( NNLayer.Activations.Memory(:,tt,:) , false , fcn_types.(NNLayer.ActFcn) ) .* permute( Actvfcn( permute( NNLayer.Activations.GatesIn.output(:,tt,:) , [1 3 2] )  , true , fcn_types.sigmoid ), [1 3 2] );
                        
                        
                        if tt~=1
                            delout(:,tt-1,:) = permute( NNLayer.Weights.wrec.forget * permute( d_f(:,tt,:) ,[1 3 2])' ...
                                + NNLayer.Weights.wrec.input * permute( d_i(:,tt,:) ,[1 3 2])' ...
                                + NNLayer.Weights.wrec.activate * permute( d_a(:,tt,:) ,[1 3 2])'  ...
                                + NNLayer.Weights.wrec.output * permute( d_o(:,tt,:) ,[1 3 2])' ,[2 3 1]);
                            
                        elseif nargout>2 % because the backpropogated reccurent input state & cell state is required.
                            del_recc_BP = permute( NNLayer.Weights.wrec.forget * permute( d_f(:,tt,:) ,[1 3 2])' ...
                                + NNLayer.Weights.wrec.input * permute( d_i(:,tt,:) ,[1 3 2])'  ...
                                + NNLayer.Weights.wrec.activate * permute( d_a(:,tt,:) ,[1 3 2])'  ...
                                + NNLayer.Weights.wrec.output * permute( d_o(:,tt,:) ,[1 3 2])'  ,[2 3 1]);
                            
                        end
                        
                        d_x(:,tt,:) = permute( NNLayer.Weights.wIN.forget * permute( d_f(:,tt,:) ,[1 3 2])' ...
                            + NNLayer.Weights.wIN.input * permute( d_i(:,tt,:) ,[1 3 2])' ...
                            + NNLayer.Weights.wIN.activate * permute( d_a(:,tt,:) ,[1 3 2])'  ...
                            + NNLayer.Weights.wIN.output * permute( d_o(:,tt,:) ,[1 3 2])' , [2 3 1] );
                        
                        
                        % Store Weight Error Gradients
                        for smp = 1:size(Xin_t,1)
                            % Input Weights gate updates - [Prior Nodes x Current Nodes x Samples]
                            NNLayer.dEdW.wIN.forget(:,:,smp)     = NNLayer.dEdW.wIN.forget(:,:,smp) + Xin_t(smp,:)' * permute( d_f(smp,tt,:) ,[1 3 2]) ; % dEdW.wIN.forget(:,:,smp) + Xin_t(smp,:)' * permute( d_f(smp,tt,:) ,[1 3 2]) ;
                            NNLayer.dEdW.wIN.input(:,:,smp)      = NNLayer.dEdW.wIN.input(:,:,smp) + Xin_t(smp,:)' * permute( d_i(smp,tt,:) ,[1 3 2]) ;
                            NNLayer.dEdW.wIN.activate(:,:,smp)   = NNLayer.dEdW.wIN.activate(:,:,smp) + Xin_t(smp,:)' * permute( d_a(smp,tt,:) ,[1 3 2]);
                            NNLayer.dEdW.wIN.output(:,:,smp)     = NNLayer.dEdW.wIN.output(:,:,smp) + Xin_t(smp,:)' * permute( d_o(smp,tt,:) ,[1 3 2]);
                            
                            if tt < NNLayer.Nstates
                                % Reccurent Unit Weights gate updates
                                NNLayer.dEdW.wrec.forget(:,:,smp)	= NNLayer.dEdW.wrec.forget(:,:,smp) + permute( NNLayer.Activations.HiddenOut(smp,tt,:) ,[1 3 2])' * permute( d_f(smp,tt+1,:) ,[1 3 2]); % dEdW.wrec.forget(:,:,smp) + permute( NNLayer.Activations.HiddenOut(smp,tt,:) ,[1 3 2])' * permute( d_f(smp,tt+1,:) ,[1 3 2]);
                                NNLayer.dEdW.wrec.input(:,:,smp)	= NNLayer.dEdW.wrec.input(:,:,smp) + permute( NNLayer.Activations.HiddenOut(smp,tt,:) ,[1 3 2])' * permute( d_i(smp,tt+1,:) ,[1 3 2]);
                                NNLayer.dEdW.wrec.activate(:,:,smp)	= NNLayer.dEdW.wrec.activate(:,:,smp) + permute( NNLayer.Activations.HiddenOut(smp,tt,:) ,[1 3 2])' * permute( d_a(smp,tt+1,:) ,[1 3 2]);
                                NNLayer.dEdW.wrec.output(:,:,smp)	= NNLayer.dEdW.wrec.output(:,:,smp) + permute( NNLayer.Activations.HiddenOut(smp,tt,:) ,[1 3 2])' * permute( d_o(smp,tt+1,:) ,[1 3 2]);
                            end
                            % Bias Weights gate updates
                            NNLayer.dEdW.wb.forget(:,:,smp)     = NNLayer.dEdW.wb.forget(:,:,smp) + permute( d_f(smp,tt,:) ,[2 3 1] ); % dEdW.wb.forget(:,:,smp) + permute( d_f(smp,tt+1,:) ,[2 3 1] );
                            NNLayer.dEdW.wb.input(:,:,smp)      = NNLayer.dEdW.wb.input(:,:,smp) + permute( d_i(smp,tt,:) ,[2 3 1] );
                            NNLayer.dEdW.wb.activate(:,:,smp)	= NNLayer.dEdW.wb.activate(:,:,smp) + permute( d_a(smp,tt,:) ,[2 3 1] );
                            NNLayer.dEdW.wb.output(:,:,smp)     = NNLayer.dEdW.wb.output(:,:,smp) + permute( d_o(smp,tt,:) ,[2 3 1] );
                            %                         end
                            
                            % Peephole Weights gate updates
                            if NNLayer.Peephole
                                if tt~=1
                                    NNLayer.dEdW.wpeep.forget(:,:,smp)	= NNLayer.dEdW.wpeep.forget(:,:,smp) + permute( NNLayer.Activations.Memory(smp,tt-1,:) ,[1 3 2])' * permute( d_f(smp,tt,:) ,[1 3 2]); % dEdW.wpeep.forget(:,:,smp) + permute( NNLayer.Activations.Memory(smp,tt-1,:) ,[1 3 2])' * permute( d_f(smp,tt,:) ,[1 3 2]);
                                    NNLayer.dEdW.wpeep.input(:,:,smp) 	= NNLayer.dEdW.wpeep.input(:,:,smp) + permute( NNLayer.Activations.Memory(smp,tt-1,:) ,[1 3 2])' * permute( d_i(smp,tt,:) ,[1 3 2]);
                                end
                                NNLayer.dEdW.wpeep.output(:,:,smp)   	= NNLayer.dEdW.wpeep.output(:,:,smp) + permute( NNLayer.Activations.Memory(smp,tt,:) ,[1 3 2])' * permute( d_o(smp,tt,:) ,[1 3 2]);
                            end
                        end
                        
                        if NNLayer.SelfReferencing
                            
                            % encoder-decoder model with attention; Prior Hidden
                            % out, becomes next hidden input via attention layer
                            if NNLayer.Attention
                                if tt>1 % the first timestep of the decoder takes in the initialiser token, without a context vewctor
                                    encseqlen = size(NNLayer.AttnInfo.EncoderInput,2);
                                    AttnScores = NNLayer.AttnInfo.AttentionScores(:,:,tt);
                                    DecoderIn_t_del = d_x(:,tt, end-NNLayer.Nunits+1:end ); % AttentionalDecoder is fed context part of concatenated vector: [Prior output , Context Vector (size of decoder states)]
                                    prior_Ht = NNLayer.Activations.HiddenIn(:,tt,:);
                                    [ ~ , ~ , Enc_del_t, Ht_prior_del, NNLayer] = AttentionLayerPropagate( NNLayer, prior_Ht , 'backward' , DecoderIn_t_del , AttnScores );
                                    NNLayer.AttnInfo.Enc_del_total(:,1:encseqlen,:) = (NNLayer.AttnInfo.Enc_del_total(:,1:encseqlen,:) + Enc_del_t); % [Batches x InputSeqLength x Features]
                                    NNLayer.AttnInfo.Dec_del_total(:,tt,:) = Ht_prior_del;
                                end
                                
                                del_attnContxt = Ht_prior_del;
                                
                                if tt > 1
                                    delout(:,tt-1,:) = delout(:,tt-1,:) + del_attnContxt; % attention layer concatenates with the prior Decoder output, to form the Decoder input vector. Backproagate the error from the decoder input, to the prior decoder output.
                                end
                                
                                % Xinput to LSTM is concatenated Embedded
                                % vector, with the Context vector from
                                % attention layer. To extract the BProped
                                % error, select the indicies that are not the
                                % context vector
                                if ~isempty(NNLayer.EmbeddingLayer)
                                    x_idx = NNLayer.EmbeddingLayer.Nunits;
                                else
                                    x_idx = NNLayer.Nunits;
                                end
                            else
                                if ~isempty(NNLayer.EmbeddingLayer)
                                    x_idx = NNLayer.EmbeddingLayer.Nunits;
                                else
                                    x_idx = NNLayer.Nunits;
                                end
                            end
                            
                            % this layer produces a context vector, to pass into the next self referencing layer (decoder)
                            % Pass back the error from the cell state of the LSTM layer in layer+1
                            
                            if ~isempty(NNLayer.EmbeddingLayer)
                                % BProp LSTM error to the embedding layer
                                [NNLayer.EmbeddingLayer, delXemb] = NNPropogate( NNLayer.EmbeddingLayer , permute( NNLayer.EmbeddingLayerInput(:,tt,:) , [1 3 2] ) , 'backward' , fcn_types , permute( d_x(:,tt,1:x_idx) ,[1 3 2] ) );
                                Emb_dEdW.wIN = Emb_dEdW.wIN + NNLayer.EmbeddingLayer.dEdW.wIN;
                                Emb_dEdW.wb = Emb_dEdW.wb + NNLayer.EmbeddingLayer.dEdW.wb;
                                if nargout>3
                                    varargout{4}(:,tt,:)=permute(delXemb,[1 3 2]);
                                end
                            end
                            
                        end
                        
                        
                    end
                    if NNLayer.SelfReferencing
                        % Time average the Error updates for the projection & embedding layer
                        NNLayer.ProjectionLayer.dEdW.wIN = Prj_dEdW.wIN;%./ NNLayer.Nstates;
                        NNLayer.ProjectionLayer.dEdW.wb = Prj_dEdW.wb;%./ NNLayer.Nstates;
                        if ~isempty(NNLayer.EmbeddingLayer)
                            NNLayer.EmbeddingLayer.dEdW.wIN = Emb_dEdW.wIN;% ./ NNLayer.Nstates;
                            NNLayer.EmbeddingLayer.dEdW.wb = Emb_dEdW.wb;% ./ NNLayer.Nstates;
                        end
                    end
                    
                    % d_x: Error to pass to earlier layers
                    varargout{1} = d_x;
                    varargout{2} = d_state(:,1,:).*NNLayer.Activations.Gates.forget(:,1,:); % USED FOR AN EARLIER LAYER as variable "delState_bp"; the error backpropogated trhrough the memory cell state
                    if nargout>2
                        varargout{3} = del_recc_BP;
                    end
                    
            end
        end
        
    case 'dense'
        
        switch propdirection
            case 'forward'
                
                [NNLayer.Activations.HiddenIn , Xin] = deal(DataIN); % [ Batches x Features/Units IN ]
                NNLayer.Activations.HiddenOut = Actvfcn( (Xin*NNLayer.Weights.wIN + NNLayer.Weights.wb) , false , fcn_types.(NNLayer.ActFcn) );
                % [ Batches x Features/Units OUT ]
            case 'backward'
                deltaErr = varargin{1}; % Error between this layers outputs, and the deeper layers target value. or this layers output and the final target output
                
                Xin = DataIN; % [ Batches x Features/Units ]
                
                if fcn_types.(NNLayer.ActFcn)==3
                    if size(NNLayer.Activations.HiddenOut,1)>1
                        delErr = zeros(size(deltaErr) ) ;
                    end
                    for btz=1:size(NNLayer.Activations.HiddenOut,1)
                        delErr(btz,:) = deltaErr(btz,:)*Actvfcn( (Xin( btz,:)*NNLayer.Weights.wIN + NNLayer.Weights.wb) , true , fcn_types.(NNLayer.ActFcn) ); % for softmax derivative
                    end
                else
                    delErr = Actvfcn( (Xin*NNLayer.Weights.wIN + NNLayer.Weights.wb) , true , fcn_types.(NNLayer.ActFcn) ).*deltaErr;
                end
                
                % Store Weight Error Gradients
                for smp = 1:size(Xin,1)
                    % Input Weights gate updates - [Prior Nodes x Current Nodes]
                    %                     if fcn_types.(NNLayer.ActFcn)==3
                    %                         NNLayer.dEdW.wIN(:,:,smp) = Xin(smp,:)*delErr(smp,:);
                    %                     else
                    NNLayer.dEdW.wIN(:,:,smp) = Xin(smp,:)'*delErr(smp,:);
                    %                     end
                    NNLayer.dEdW.wb(:,:,smp) = delErr(smp,:);
                end
                
                % d_x: Error to pass to earlier layers
                d_x = delErr(:,:)*NNLayer.Weights.wIN'; % [Batches x input Features]
                varargout{1} = d_x;
                
        end
        
    case 'attention'
        
        switch propdirection
            case 'forward'
                
                Xin = DataIN; % [ Batches x input elements x Features/Units IN ]
                NNLayer.Activations.HiddenOut = Actvfcn( (Xin*NNLayer.Weights.wIN + NNLayer.Weights.wb) , false , fcn_types.(NNLayer.ActFcn) );
                % [ Batches x Features/Units OUT ]
            case 'backward'
                deltaErr = varargin{1}; % Error between this layers outputs, and the deeper layers target value. or this layers output and the final target output
                
                Xin = DataIN; % [ Batches x Features/Units ]
                
                delErr = Actvfcn( (Xin*NNLayer.Weights.wIN + NNLayer.Weights.wb) , true , fcn_types.(NNLayer.ActFcn) ).*deltaErr;
                if fcn_types.(NNLayer.ActFcn)==3
                    delErr = deltaErr*Actvfcn( (Xin*NNLayer.Weights.wIN + NNLayer.Weights.wb) , true , fcn_types.(NNLayer.ActFcn) ); % for softmax derivative
                end
                
                % Store Weight Error Gradients
                for smp = 1:size(Xin,1)
                    % Input Weights gate updates - [Prior Nodes x Current Nodes]
                    %                     if fcn_types.(NNLayer.ActFcn)==3
                    %                         NNLayer.dEdW.wIN(:,:,smp) = Xin(smp,:)*delErr(smp,:);
                    %                     else
                    NNLayer.dEdW.wIN(:,:,smp) = Xin(smp,:)'*delErr(smp,:);
                    %                     end
                    NNLayer.dEdW.wb(:,:,smp) = delErr(smp,:);
                end
                
                % d_x: Error to pass to earlier layers
                d_x = delErr(:,:)*NNLayer.Weights.wIN'; % [Batches x input Features]
                varargout{1} = d_x;
                
        end
        
        
    case 'Convolutional'
        switch propdirection
            case 'forward'
                
            case 'backward'
                
        end
        
end

end


function [NNLayer] = LSTMCell( MemorycellOut_Last , Xin , Hin , Weights , fcn_types , NNLayer , tt , smpl )
% Xin, Hin: [batches x states]
% Xin_FIinO = repmat( Xin, [4 4] )*[ Weights.wIN.forget ; Weights.wIN.input ; Weights.wIN.activate ; Weights.wIN.output ];
% Z0 = repmat( zeros(size(Weights.wIN.forget)) , 4,4 );

% GatesInput / Pre-Activation
NNLayer.Activations.GatesIn.forget(smpl,tt,:) = (Xin * Weights.wIN.forget ) + (Hin * Weights.wrec.forget ) + Weights.wb.forget;
NNLayer.Activations.GatesIn.input(smpl,tt,:) = (Xin * Weights.wIN.input ) + (Hin * Weights.wrec.input ) + Weights.wb.input;
if NNLayer.Peephole
    NNLayer.Activations.GatesIn.forget(smpl,tt,:) = NNLayer.Activations.GatesIn.forget(smpl,tt,:) + (MemorycellOut_Last * Weights.wpeep.forget );
    NNLayer.Activations.GatesIn.input(smpl,tt,:) = NNLayer.Activations.GatesIn.input(smpl,tt,:) + (MemorycellOut_Last * Weights.wpeep.input );
end

NNLayer.Activations.GatesIn.activate(smpl,tt,:) = (Xin * Weights.wIN.activate ) + (Hin * Weights.wrec.activate ) + Weights.wb.activate;


% Peephole LSTM
% ForgetgateOut  = Actvfcn( (Xin * Weights.wIN.forget ) + (Hin * Weights.wrec.forget ) + (MemorycellOut_Last * Weights.wpeep.forget ).*YP + Weights.wb.forget , false , fcn_types.sigmoid ); % peephole: memory cell going into forget, input and output gates.
NNLayer.Activations.Gates.forget(smpl,tt,:)  = Actvfcn( permute( NNLayer.Activations.GatesIn.forget(smpl,tt,:) , [1 3 2] ) , false , fcn_types.sigmoid ); % peephole: memory cell going into forget, input and output gates.
% ( input gate / Update gate )
% InputgateOut = Actvfcn( (Xin * Weights.wIN.input ) + (Hin * Weights.wrec.input )  + (MemorycellOut_Last * Weights.wpeep.input ).*YP + Weights.wb.input , false , fcn_types.sigmoid ); % peephole: memory cell going into forget, input and output gates.
NNLayer.Activations.Gates.input(smpl,tt,:) = Actvfcn( permute( NNLayer.Activations.GatesIn.input(smpl,tt,:) , [1 3 2] ) , false , fcn_types.sigmoid ); % peephole: memory cell going into forget, input and output gates.

% inputactivation = Actvfcn( (Xin * Weights.wIN.activate ) + (Hin * Weights.wrec.activate ) + Weights.wb.activate , false , fcn_types.(NNLayer.ActFcn)  );
NNLayer.Activations.Gates.activate(smpl,tt,:) = Actvfcn( permute( NNLayer.Activations.GatesIn.activate(smpl,tt,:) , [1 3 2] ) , false , fcn_types.(NNLayer.ActFcn)  );

% MemorycellOut = ForgetgateOut .* MemorycellOut_Last ... % f(j).*state(j-1)
%     + InputgateOut .* inputactivation; % i(j).*a(j)
NNLayer.Activations.Memory(smpl,tt,:) = NNLayer.Activations.Gates.forget(smpl,tt,:) .* permute( MemorycellOut_Last , [1 3 2] ) ... % f(j).*state(j-1)
    + NNLayer.Activations.Gates.input(smpl,tt,:) .* NNLayer.Activations.Gates.activate(smpl,tt,:); % i(j).*a(j)

% OutputgateOut = Actvfcn( (Xin * Weights.wIN.output ) + (Hin * Weights.wrec.output ) + (MemorycellOut * Weights.wpeep.output ).*YP + Weights.wb.output , false , fcn_types.sigmoid ); % peephole: memory cell going into forget, input and output gates.
NNLayer.Activations.GatesIn.output(smpl,tt,:) = (Xin * Weights.wIN.output ) + (Hin * Weights.wrec.output ) + Weights.wb.output;
if NNLayer.Peephole
    NNLayer.Activations.GatesIn.output(smpl,tt,:) = NNLayer.Activations.GatesIn.output(smpl,tt,:) + ( permute(NNLayer.Activations.Memory(smpl,tt,:), [1 3 2] ) * Weights.wpeep.output );
end

NNLayer.Activations.Gates.output(smpl,tt,:) = Actvfcn( permute( NNLayer.Activations.GatesIn.output(smpl,tt,:) , [1 3 2] ) , false , fcn_types.sigmoid ); % peephole: memory cell going into forget, input and output gates.

% hiddenOut = OutputgateOut .* Actvfcn( MemorycellOut , false , fcn_types.(NNLayer.ActFcn) );
NNLayer.Activations.HiddenOut(smpl,tt,:) = permute( NNLayer.Activations.Gates.output(smpl,tt,:), [1 3 2] ) .* Actvfcn( permute(NNLayer.Activations.Memory(smpl,tt,:), [1 3 2] ) , false , fcn_types.(NNLayer.ActFcn) );

% gates = struct('forget', ForgetgateOut,'input', InputgateOut,'output',OutputgateOut ,'activate',inputactivation );
%                         NNLayer.Activations.Memory(smpl,tt,:) = memcellOut;
% %                         NNLayer.Activations.HiddenOut(smpl,tt,:) = hiddenOut;
%                         NNLayer.Activations.Gates.forget(smpl,tt,:) = gates.forget;
% %                         NNLayer.Activations.Gates.input(smpl,tt,:) = gates.input;
% %                         NNLayer.Activations.Gates.output(smpl,tt,:) = gates.output;
%                         NNLayer.Activations.Gates.activate(smpl,tt,:) = gates.activate;
end

% Xin[b x m] * Wf[m x n] = [b x n]
% Xin[b x m] * Wi[m x n] = [b x n]
% Xin_FIinO = repmat( Xin, [4 1] )*[ Weights.wIN.forget ; Weights.wIN.input ; Weights.wIN.output ] ;%   [4b by 4m] x [ Wf , repmat( zeros(size(Weights.wIN.forget)) , 4,4 )
% Weights.wIN.activate ;

function [outV] = Actvfcn( inV , deriv , act_type , varargin )
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
        
        %             norm_const = -max(inV);
        outV = tanh(inV);%exp(inV+norm_const)/sum(exp(inV+norm_const));
        if ~deriv
        else % compute derivative of activation function
            if numel(varargin)>0
                prior_outV = varargin{1};
                outV = 1 - (prior_outV).^2; %1 - tanh(inV).^2;
            else
                outV = 1 - (outV).^2; %1 - tanh(inV).^2;
            end
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
        norm_const = -max(inV , [] , 2); % [ Batches x Logit scores (features) ]
        Z = inV+norm_const;
        Z = inV-max(inV,[],2);
        outV = exp(Z) ./ sum( exp(Z) , 2 );%             outV = sinh(inV) ./ cosh(inV);
        if ~deriv
            
        else % compute derivative of activation function
            softmax = outV;%exp( Z ) ./ sum( exp( Z ) , 2);%             outV = sinh(inV) ./ cosh(inV);
            outV = diag(softmax) - repmat( softmax' , 1 , numel(inV) ).*repmat( softmax , numel(inV) , 1 );
        end
        if any(isnan(outV)) , keyboard , end
    case 6%'linear' % nonlin_1 = @(x,deriv) strcmp(deriv,'true').*((x>0).*1 + (x<=0).*0) + strcmp(deriv,'false').*max(0,x);
        if ~deriv
            outV = inV;
        else % compute derivative of activation function
            outV = 1;
        end
        
end
% if any(isnan(outV)) , keyboard , end
end