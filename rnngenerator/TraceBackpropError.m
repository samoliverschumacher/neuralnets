function [StoreGrads] = TraceBackpropError(NNModels,dCda,X_input,fcn_types)

outseqlen = NNModels{end}.Nstates;
inseqlen = NNModels{end-1}.Nstates;

b = 1:size(NNModels{end}.XInput,1);
NNModels{end-1}.curStep = size(NNModels{end-1}.XInput,2);
NNModels{end}.curStep = size(dCda,2);


%% 1 - Post Decoder Errors
        % Prediction Error
        StoreGrads.Dec.PrdErr = dCda;
        if any(contains( fieldnames(NNModels{end}) , 'ProjectionLayer'))
            for td=1:outseqlen
                    % HiddenState Out Error
                    [~, delXPrj] = NNPropogate( NNModels{end}.ProjectionLayer , permute(NNModels{end}.Activations.HiddenOut(:,td,:),[1 3 2]) , 'backward' , fcn_types , permute(dCda(:,td,:),[1 3 2]));
                    StoreGrads.Dec.HoutErr(b,td,:) = permute(delXPrj(b,:),[1 3 2]);
            end
        else , StoreGrads.Dec.HoutErr = NaN;
        end
%% 2 - Pre Decoder Errors
        % Memory In Error
        [~, delX, delMemIn, delHin, delXemb] = NNPropogate( NNModels{end} , NNModels{end}.XInput , 'backward' , fcn_types , dCda );
        StoreGrads.Dec.MemInErr = delMemIn;
        % HiddenState In Error
        StoreGrads.Dec.HinInErr = delHin;
        % Embedded Prediction In Error (Xinput 1)
        StoreGrads.Dec.EmbPrdErr = delXemb;
        if NNModels{end}.Attention
            % Context Error (Xinput 2)
            StoreGrads.Dec.CtxErr = delX(:,:, end-NNModels{end}.Nunits+1:end);
    %% 3 - Pre Attention Errors
            % Attention HiddenState In Error (Attn Input 1)
            StoreGrads.Dec.AttnHinError = NNModels{end}.AttnInfo.Dec_del_total;
            % Attention EncoderSequence In Error (Attn Input 2)
            StoreGrads.Dec.AttnEncError = NNModels{end}.AttnInfo.Enc_del_total;
        else
            StoreGrads.Dec.CtxErr = NaN; StoreGrads.Dec.AttnHinError = NaN; StoreGrads.Dec.AttnEncError = NaN;
        end
        if NNModels{end}.Attention
            delHEnc = StoreGrads.Dec.AttnEncError; 
            delHEnc(:,end,:) = delHEnc(:,end,:) + delHin;
        else
            delHEnc = zeros(size(NNModels{end-1}.Activations.HiddenOut));
            delHEnc(:,NNModels{end-1}.curStep,:) = delHin;
        end
%% 4 - Encoder Errors
        % Memory In Error
        [~, delX_Enc, delMemIn_Enc, delHin_Enc] = NNPropogate( NNModels{end-1} , NNModels{end-1}.XInput , 'backward' , fcn_types , delHEnc , delMemIn , delHin );
        StoreGrads.Enc.MemInErr = delMemIn_Enc;
        % HiddenState In Error
        StoreGrads.Enc.HinInErr = delHin_Enc;
        % Xinput Error
        StoreGrads.Enc.XinErr = delX_Enc;
%% 5 - Embedding Error
for te=1:size(X_input,2)
        %Encoder Embedding Error
        [~ , delX] = NNPropogate( NNModels{1} , permute( X_input(:,te,:) ,[1 3 2] ) , 'backward' , fcn_types , permute( delX_Enc(:,te,:) ,[1 3 2] ) );
        StoreGrads.Enc.EncEmbErr(:,te,:) = permute(delX,[1 3 2]);
end


BPseq=[];
BPseq = [BPseq, norm(StoreGrads.Dec.PrdErr(:)) ./numel(StoreGrads.Dec.PrdErr(:))]; 
BPseq = [BPseq, norm(StoreGrads.Dec.HoutErr(:)) ./numel(StoreGrads.Dec.HoutErr(:))];

BPseq = [BPseq, norm(StoreGrads.Dec.MemInErr(:)) ./numel(StoreGrads.Dec.MemInErr(:))];
BPseq = [BPseq, norm(StoreGrads.Dec.HinInErr(:)) ./numel(StoreGrads.Dec.HinInErr(:))];
BPseq = [BPseq, norm(StoreGrads.Dec.EmbPrdErr(:)) ./numel(StoreGrads.Dec.EmbPrdErr(:))];
BPseq = [BPseq, norm(StoreGrads.Dec.CtxErr(:)) ./numel(StoreGrads.Dec.CtxErr(:))];

BPseq = [BPseq, norm(StoreGrads.Dec.AttnHinError(:)) ./numel(StoreGrads.Dec.AttnHinError(:))];
BPseq = [BPseq, norm(StoreGrads.Dec.AttnEncError(:)) ./numel(StoreGrads.Dec.AttnEncError(:))];

BPseq = [BPseq, norm(StoreGrads.Enc.MemInErr(:)) ./numel(StoreGrads.Enc.MemInErr(:))];
BPseq = [BPseq, norm(StoreGrads.Enc.HinInErr(:)) ./numel(StoreGrads.Enc.HinInErr(:))];
BPseq = [BPseq, norm(StoreGrads.Enc.XinErr(:)) ./numel(StoreGrads.Enc.XinErr(:))];
BPseq = [BPseq, norm(StoreGrads.Enc.EncEmbErr(:)) ./numel(StoreGrads.Enc.EncEmbErr(:))];

StoreGrads.lbls = [strcat("Dec-", fieldnames(StoreGrads.Dec)) ; strcat("Enc-", fieldnames(StoreGrads.Enc))];

StoreGrads.BPseq = BPseq;

end