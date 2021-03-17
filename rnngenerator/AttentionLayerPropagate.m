function [ContextV, AttnScore, varargout] = AttentionLayerPropagate( NNLayer , H_tprior, NNDirection , varargin )
EncV = NNLayer.AttnInfo.EncoderInput;
encseqlen = size(EncV,2);
if NNDirection=="forward"    % Feed Forward
    % g(x) = scores*
    AttnScore = NaN( size(H_tprior,1) , encseqlen );
    switch NNLayer.AttnInfo.ScoringFcn 
        case "general"

            DVec = NNLayer.AttnInfo.Weights * H_tprior';
            for btc = 1:size(EncV,1)
                AttnScore(btc,:) = (permute( EncV(btc , : , :), [2 3 1] ) * DVec(:,btc) )';
            end
        
        case "dotproduct"
        
            for btc = 1:size(EncV,1)
                AttnScore(btc,:) = (permute( EncV(btc , : , :), [2 3 1] ) * H_tprior(btc,:)' )';
            end
            
    end
    sftmxID = 3; % fcn_types.softmax
    alpha = Actvfcn( AttnScore , false , sftmxID );
    ContextV = NaN( size(EncV,1) , size(EncV,3) );
    for btc = 1:( size(H_tprior,1) )
        ContextV(btc,:) = alpha(btc,:)*permute( EncV(btc,:,:) , [2 3 1] );
    end

    % context = softmax( Enc*Hprior )*Enc
elseif NNDirection=="backward"    % Backwards Propogate
        %% BACKPROPOGATE ATTENTION LAYER
    % OUTPUTS
    % Enc_del_t = dLoss_dContext .* dContext_dEncoder;
    % Ht_prior_del = dLoss_dContext .* dContext_dalpha .* dalpha_dScore .* dScore_dHoutPrior
    % 
    % INPUTS
    % ContextVect_del = Decoder_del_t(:,:,EncoderNunits+1:end); (dL/dContextVect)
    % AttnScore
    ContextV=[];
    DecoderIn_t_del = varargin{1}; % error propogated into this time step from a future timestep
    AttnScore = varargin{2}; % attention score evaluated for this time step
    AttnScore = AttnScore( : , 1:encseqlen );
    
    sftmxID = 3; % fcn_types.softmax
    
    dLoss_dContext = DecoderIn_t_del;%( : , : , 1:EncoderNunits ); % [ batches x t=1 x features ]
    
    alpha = Actvfcn( AttnScore , false , sftmxID );
    dContext_dEncoder = alpha;
        
    % gradient back to Encoder, due to timestep "t" in output sequence
    Enc_del1_t = dLoss_dContext .* dContext_dEncoder;
    
    seqinlen = size(EncV,2); % source sequence in length
    bsz = size(EncV,1); % batch size
    
        % dc_dB: [s x s x t]
        dc_dB = 1;%ones( size(H_tprior,3) , size(EncV,2) ); % ones( size(H_tprior,3) , size(H_tprior,3) , size(H_tprior,2) );
        
        if NNLayer.AttnInfo.ScoringFcn=="general"
            % W: [s_enc x s_dec]
            W = NNLayer.AttnInfo.Weights;
            dEdW = zeros( [size(W) , bsz] );
            dEdWb_t_ini = zeros(size(W));
        end
        
        del_sc = zeros( bsz, seqinlen );

        % D: [s]    (decoder values)
            D = permute(H_tprior ,[3 2 1]);
            % E: [s,t]
            E = permute(EncV,[3 2 1]);
            % delalpha        
            delalpha = sum( DecoderIn_t_del.*EncV , 3);
            
            for b=1:bsz
                % dalph_dsc: [t x t]
                dalph_dsc = Actvfcn( AttnScore(b,:) , true , sftmxID );
                del_sc( b , 1:seqinlen) = delalpha(b,:) * dalph_dsc;
                % ddhat_dW: [s_dec x s_dec]
                if NNLayer.AttnInfo.ScoringFcn=="general"
                    ddhat_dW = D(:,1,b)';
                    % initialise the time accumulation of weight gradients
                    dEdWb_t = dEdWb_t_ini;
                    for t=1:seqinlen
                        % dsc_ddhat: [t x s_enc]  |  dsc_ddhat_t: [s_enc]   |  [t,s]*[s,1] --> [t,1] [t]
                        dsc_ddhat = E(:,t,b);
                        % dsc_dW: [t x s_enc x s_dec]  |  dsc_dW_t: [s_enc x s_dec]
                        dsc_dW = dsc_ddhat * ddhat_dW;
                        
                        dEdWb_t = dsc_dW.*permute(del_sc(b,t),[1 3 2]) + dEdWb_t;
                    end
                    % dEdW is the sum across timesteps
                    dEdW(:,:,b) = dEdWb_t;
                end
            end
            
            if NNLayer.AttnInfo.ScoringFcn=="general"
                dsc_denc2 = W*permute(D(:,1,:),[1 3 2] );
                % add this decoder time step's error updates to the stored values for this backprop pass
                NNLayer.AttnInfo.dEdW = dEdW + NNLayer.AttnInfo.dEdW;
                Ht_prior_del = [ 0 ]; %zeros( size(EncV(:,1,:) ) );
                for t=1:seqinlen
                    Ht_prior_del = permute( del_sc(:,t).*permute(E(:,t,:),[1 3 2] )'*W ,[1 3 2] ) + Ht_prior_del;
                end
            else
                dsc_denc2 = permute(D(:,1,:),[1 3 2] );
                Ht_prior_del = [ 0 ]; %zeros( size(EncV(:,1,:) ) );
                for t=1:seqinlen
                    Ht_prior_del = permute( del_sc(:,t).*permute(E(:,t,:),[1 3 2] )' ,[1 3 2] ) + Ht_prior_del;
                end
            end
            
            Enc_del2_t = del_sc.*permute( dsc_denc2,[2 3 1] );
            
            % sum across the entire input sequence Scores before propogating back


    Enc_del_t = Enc_del2_t + Enc_del1_t;
    % gradient back to Decoders hidden state at timestep "t-1"
%     Ht_prior_del = dLoss_dContext .* dContext_dalpha .* dalpha_dScore .* dScore_dHoutPrior;
    
    varargout{1} = Enc_del_t; % does the attention layer feed back into the Decoder at t-1 ?
    varargout{2} = Ht_prior_del; % does the attention layer feed back into the Decoder at t-1 ?
    varargout{3} = NNLayer;

end
end