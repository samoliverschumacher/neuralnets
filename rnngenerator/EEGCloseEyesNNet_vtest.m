
load('EEGEyeStateWorkspace.mat','windowlength','TimeSeries','idx_blink','idx_openclose','N','Nvars','win_trainN','N_seconds');
% remove 'eyes blinking' from the dataset: closed eyes for under 1 second only
blinkduration_seconds = 0.45;
blinkduration_tsteps = N*(blinkduration_seconds/N_seconds);
idx_blink = [movsum( idx_openclose ,[blinkduration_tsteps blinkduration_tsteps])]<blinkduration_tsteps & idx_openclose;
idx_openclose = ~idx_blink & idx_openclose;

% downsample observations to the average of a 3 step window
vars=[ [5 , 8] , [1 2 11 13] ];%[1 2 11 13];%[ [5 , 8] , [1 2 11 13] ];%[1:14];%[ [5 , 8] , [1 2 11 13] ];%[13 8 11 1 2];%(1:14);%[13 8 11 1 2];%(1:14); %[13 8 11 1 2]; % interesting Variables % [1 2 13 14];
dsr = 3; % downsampling ratio
dwn_TS = zeros( floor(N/3)-1 , Nvars+1 );
for sbp=vars
    dwn_TS(:,sbp) = arrayfun(@(ii) mean([TimeSeries(ii-(dsr-1):ii+(dsr-1),sbp)]) , [dsr:dsr:numel(TimeSeries(:,1))-(dsr-1)]' );
end
dwn_TS(:,15) = arrayfun(@(ii) mean([idx_openclose(ii-(dsr-1):ii+(dsr-1), 1 )]) , [dsr:dsr:numel(TimeSeries(:,1))-(dsr-1)]' );
dwn_TS(:,15) = round(dwn_TS(:,15));

ids = [dsr:dsr:numel(TimeSeries(:,1))-(dsr-1)]';
Rdwn( ids-2 ,1 ) = dwn_TS( : ,1);

% reshape the downsampled EEGs: [timesteps, (none) , variable]
Ts_X = permute( dwn_TS( : , vars) , [1 3 2] );
% for each variable, calculate the moving sum of differences, and then
% normalize into the range [0 , 1]
for ii=1:size(Ts_X,3) 
    Xdiff = [0; diff(Ts_X(:,:,ii)) ];
    Ts_X(:,:,ii) =  movsum( Xdiff,[3 0]);
    Ts_X(:,:,ii) = normalize( Ts_X(:,:,ii) ,1,'range',[-1 1]);
end

% plot differneced normalised variable with/without moving sum applied.
plot([Ts_X(:,1,ii), normalize(Xdiff,1,'range',[-1 1]) ])
% xlim([2000 2100])
legend('Normalised moving sum of differenced #EEG 2','Normalised differenced #EEG 2')
xlabel('timestep');
title('3-step Moving sums effect on EEG')

Ts_Y = permute( dwn_TS( : , end ) , [1 3 2]);
Ts_Y = permute(dummyvar(categorical(Ts_Y)) , [1 3 2]);

% create minibatches of the sequences of variables.
seqlen_in = 40; % input sequence length
seqlen_out=5; % target sequence length
Nbatches = 12; % number of batches in each training epoch
[DataX_batched, DataY_batched, Xind_c, Yind_c, singleseqlen] = batchdatasets( Nbatches, Ts_X , Ts_Y , 'seq2batch' ,seqlen_in, seqlen_out);

% size of each minibatch input and output
InputDataSize = size(DataX_batched{1});
OutputDataSize = size(DataY_batched{1});
batchsize = size(DataX_batched{1},1);

disp( batchsize )
disp(['Input sequences'; DataX_batched ])
disp(['1st batch of Target sequences'; DataY_batched(1) ])

% Plot input-output pairs from the batched data
% Find eye-state transition timesteps
eyeOid = find([ 0 ;diff( Ts_Y(:,1,2),1,1)]==1);
eyeCid = find([ 0 ;diff( Ts_Y(:,1,2),1,1)]==-1);

% find the sample whose first prediction timestep is at the first Close to
% Open eye transtion
smplB = find(Yind_c(:,1)==eyeOid(1),1) - 2;
% sample 20 steps prior
smplA = smplB-20;
% plot EEGs at sample A & B
figure;
pEA = plot( Xind_c( smplA ,: ) , squeeze(DataX_batched{1}( smplA ,:,:)),'b','LineWidth',1.5); hold on;
pEB = plot( Xind_c( smplB ,: ) , squeeze(DataX_batched{1}(smplB,:,:)),'r.-');
% Plot target sequence following input samples.
pSA = plot(Yind_c( smplA ,:), squeeze(DataY_batched{1}(smplA,:,2)),'b-d','MarkerFaceColor','b');
pSB = plot(Yind_c( smplB ,:), squeeze(DataY_batched{1}(smplB,:,2)),'r-d','MarkerFaceColor','r');
xlabel('timestep'); ylabel('Input EEG value'); title('Input-Output pairs from training minibatch #1')
legend([pEA(1),pEB(1),pSA(1),pSB(1)],'EEGs @ sample sequence A','EEGs @ sample sequence B','Eye state @ Seq A','Eye state @ Seq B')


%% Topology
    epochs = 800;
%     GDOptimizer = 'Adam';
%     learnrate = 0.005;
%     LossType = "MultiClassCrossEntropy";%"MultiClassCrossEntropy";%"WeightedBinaryClassification";
    
    trainingSettings = struct();
    trainingSettings.LossType = "MultiClassCrossEntropy";%"MultiClassCrossEntropy";%"WeightedBinaryClassification";
    trainingSettings.learnrate = 0.005;
    trainingSettings.GDOptimizer = 'Adam';
    trainingSettings.gradclip = true;
    
    % input embedding
    encoderHUs = 45;
    decoderHUs = 45;
    classifierHUs = OutputDataSize(3);
    
    % create Encoder
    [NNLayerEnc] = GenerateNNetLayer( encoderHUs , InputDataSize(1) , InputDataSize(3) , "LSTM" , "tanh" ...
        , InputDataSize(2) , struct('resetstate',false,'predictsequence',false ) );
    % create Decoder
    [NNLayerDec] = GenerateNNetLayer( decoderHUs , InputDataSize(1) , NNLayerEnc.Nunits , "LSTM" , "tanh" ...
        , OutputDataSize(2) , struct('resetstate',false,'predictsequence',true ) );
    
    % Projection Layer to Output tokens
    [NNLayerFinal] = GenerateNNetLayer( classifierHUs , OutputDataSize(1) , NNLayerDec.Nunits , "dense" , "softmax" );
    
    NNModels = [{NNLayerEnc},{NNLayerDec},{NNLayerFinal}];
    

disp([newline ,'Input Layer (Encoder); ']) , disp( NNModels{1} )
disp([newline ,'Prediction Layer (Decoder); ']) , disp( NNModels{2} )
disp([newline ,'Final classification Layer; ']) , disp( NNModels{3} )

%% Train the Neural Net

    [NNModels, ~, ~ , ~ ] = ...
        TrainSequnceNNetV3(NNModels, epochs, trainingSettings , DataX_batched, DataY_batched, [], []);
%%
    
%     save('EEGNNetModels.mat')

    % Run an inference only loop over the dataset with the trained model
    [~, Prediction, ErrorMetric1, ErrorMetric2] = ...
        TrainSequnceNNetV3(NNModels, epochs, trainingSettings , DataX_batched, DataY_batched, [], [],true);
    
    
    % unroll prediction, take the average of each sequence from consecutive
    % starting points
    DataY_unbatched = cat(1,DataY_batched{:}); 
    DataX_unbatched = cat(1,DataX_batched{:});
     
    disp('Average Accuracy for Predicting "Open" class across output sequence;')
    disp(mean(round(Prediction(:,:,1))==DataY_unbatched(:,:,1)))
    
    % vector containing all 5-step predictions & errors (5 steps * 4800 samples = 24000 predictions)
    PrdVect = reshape(Prediction(:,:,1), [numel(Prediction(:,:,1)) , 1] );
    TrgVect = reshape(DataY_unbatched(:,:,1), [numel(DataY_unbatched(:,:,1)) , 1] );
    ErrVect = (PrdVect - TrgVect);
    
    % average of all predictions
    PredSeqAvg = accumarray( Yind_c(:) , PrdVect ,[],@mean , NaN);
    MeanAbsErr = accumarray( Yind_c(:) , ErrVect ,[],@(x) mean(abs(x)) , NaN);
    % recreate entire Target sequence from unbatched data
    TargetSeq = accumarray( Yind_c(:) , TrgVect ,[],@mean , NaN);
    
    % transition from closed eye to open
	openeyeId = find([NaN;diff(TargetSeq )]==1);
	% transition from open eye to closed
    closeeyeId = find([NaN;diff(TargetSeq)]==-1);
    % extract mean absolute error 10 timesteps either side of a "Open eye"
    TI = 15;
    ABSE_openeye = arrayfun(@(rw) MeanAbsErr((rw-TI):(rw+TI))' ,[openeyeId] ,'un',0);
    ABSE_openeye = nanmedian(cat(1,ABSE_openeye{:}),1);
    % extract mean absolute error 10 timesteps either side of a "Close eye"
    ABSE_closeeye = arrayfun(@(rw) MeanAbsErr((rw-TI):(rw+TI))' ,[closeeyeId] ,'un',0)';
    ABSE_closeeye = nanmedian(cat(1,ABSE_closeeye{:}),1);
    
    % plot the error either side of a state transition.
    figure; 
    plot( [ABSE_openeye] );hold on; 
    plot( [ABSE_closeeye] ); ylabel('Avg Abs Error'); xlabel('Prediction Timestep'); legend({'opening eye','closing eye'})
    timelabels = [strsplit(sprintf('-%g ',[TI :-1:1]')), strsplit(sprintf('+%g ',[1:TI ]'))];
    timelabels(end)= []; timelabels(TI +1) = {'t=0'};
    xLabInds = unique([ceil(linspace(1,TI+1,4)) , ceil(linspace(TI+1, 2*TI +1,4)) ]);
    xticks(xLabInds); set(gca,'xticklabels',timelabels(xLabInds) ); xlim([1 2*TI+1]); ylim([0 1]);
    
    % view the average prediction error, across the output sequence.
%     ABSE_seqOut = arrayfun(@(rw) ABSE(rw,find( ~isnan(ABSE(rw,:)),5)) ,[1:size(ABSE,1)]','un',0);    
%     figure; bar( mean(cat(1,ABSE_seqOut{:}) ,1) ); ylabel('Avg Abs Error'); xlabel('Prediction Timestep')
    
    % plot prediction error, and key EEG values across the sequnce
    figure('Position',[141 471 1471 470]);
    subplot(2,1,1); title('Prediction Error')
    bh = bar(round(PredSeqAvg),'LineStyle','none','FaceAlpha',0.2); 
    yyaxis right;bar([MeanAbsErr]); ylim([0 1]); set(gca,'Ytick',[0 1]); 
    legend('Eye state prediction (1=Open)','Prediction error','location','north'); 
    
    subplot(2,1,2); title('EEGs (downsampled 3:1)')
    hold on; plot( normalize( movmean( dwn_TS( 1:4800 ,vars([1 2 3]) ),5 , 1), 1 ,'range')-0.5 );
    yyaxis right; bar( TargetSeq ,'LineStyle','none','FaceAlpha',0.2 ,'FaceColor',bh.FaceColor); 
    set(gca,'Ytick',[0 1])
    legend('EEG 1','EEG 2','EEG 13','Target eye state (1=Open)','location','north')
    
    % Plot prediction as time-series, and Confusion Matrix
    [ fh1 , fh2 ] = plotPrediction( round(PredSeqAvg) , TargetSeq, PredSeqAvg , TargetSeq , 117/(N) *seqlen_out*dsr );
    
 
% function [outXY] = split_sequence(Outputs , n_steps , n_steps_out)
% outXY = cell(1,2);
% [ts,vr] = size(Outputs);
% counter=0;
% for ii = 1:ts
%     end_ix = ii+ n_steps-1;
%     out_end_ix = end_ix + n_steps_out;
%     if out_end_ix > ts
%         return
%     end
% %     [seq_in , seq_out] = deal({Outputs(ii:end_ix-1)} , {Outputs(end_ix)});
%     seq_in = {Outputs(ii:end_ix, : )};
%     seq_out = {Outputs(end_ix+1:out_end_ix , : )};
%     counter = counter+1;
%     outXY(counter,:) = [seq_in , seq_out];
% end
% end

    
    
    %% EEG signals at Eye-State Transition 
Xts2 = permute( normalize(dwn_TS( : , vars) ,1,'range'), [1 3 2] );
% eyes opening indices
eyeOid = find([NaN;diff( Ts_Y(:,1,2),1,1)]==1);
eyeCid = find([NaN;diff( Ts_Y(:,1,2),1,1)]==-1);

% transition interval
TI = 20;
eyeOseq = NaN(numel(eyeOid) ,2*TI +1, numel(vars));
eyeCseq  = NaN(numel(eyeCid) ,2*TI +1, numel(vars));
for vi=1:numel(vars)
    for rwop = 1:numel(eyeOid)
        seqid = (eyeOid(rwop)-TI ):(eyeOid(rwop)+TI );
        eyeOseq(rwop, seqid>0 & seqid<=size(Ts_Y,1) ,vi) = Xts2(seqid(seqid>0 & seqid<=size(Ts_Y,1)),1,vi);
    end
    for rwcl = 1:numel(eyeCid)
        seqid = (eyeCid(rwcl)-TI ):(eyeCid(rwcl)+TI );
        eyeCseq(rwcl, seqid>0 & seqid<=size(Ts_Y,1) ,vi) = Xts2(seqid(seqid>0 & seqid<=size(Ts_Y,1)),1,vi);
    end
end

% Visualise EEGs state transitions
figure('Position',[141,122,1247,819]);
timelabels = [strsplit(sprintf('-%g ',[TI :-1:1]')), strsplit(sprintf('+%g ',[1:TI ]'))];
timelabels(end)= []; timelabels(TI +1) = {'t=0'};
xLabInds = unique([ceil(linspace(1,TI+1,4)) , ceil(linspace(TI+1, 2*TI +1,4)) ]);

for V=1:numel(vars)
    subplot(3,2,V)
    pC = plot( eyeCseq(:,:,V)' ,'r' ); hold on; 
    pO = plot( eyeOseq(:,:,V)' ,'b' ); legend([pC(1),pO(1)],'Eye Closing','Eye Opening')

    xticks(xLabInds); xticklabels( timelabels(xLabInds) ); xlim([1 2*TI+1]); ylim([0 1]);
    grid minor;
    title(['EEG #',num2str(V)]); 
    xlabel('Eye-state transition window'); ylabel('Normalised EEG value')
end

%% cost asymmetry
% Taking a power of prediction for True class targets
% Loss_delta_w1 = @(Pred,Real) (Real==1).*((-Real./(Pred.^(1.5))) + ( (1-Real)./(1-(Pred.^(1.5))) )) ...
% + (Real==0).*((-Real./Pred) + ( (1-Real)./(1-Pred) ));
% % Multiplying True class error gradient by 3
% Loss_delta_w2 = @(Pred,Real) 3.*( (Real==1).*((-Real./(Pred)) + ( (1-Real)./(1-(Pred)) )) )...
% + (Real==0).*((-Real./Pred) + ( (1-Real)./(1-Pred) )) ;
% 
% figure;
% subplot(1,2,1);
% grad_Target1 = abs( Loss_delta_w1( fliplr(linspace(0.01,0.99,100)) , ones(1,100) ) ); 
% grad_Target0 = abs( Loss_delta_w1( linspace(0.01,0.99,100) , zeros(1,100) ) );
% plot( [grad_Target1 ;grad_Target0]' );
% 
% legend({'target class=1','target class=0'}); title('Prediction^(1.5) | Target==True')
% xlabel('Predicted Probability of Misclassification'); xticklabels(string([get(gca,'Xtick')]/100)); 
% ylabel('Loss'); ylim([0 , 5*median( [(grad_Target1),(grad_Target0)] ) ])
% 
% subplot(1,2,2)
% grad_Target1 = abs( Loss_delta_w2( fliplr(linspace(0.01,0.99,100)) , ones(1,100) ) ); 
% grad_Target0 = abs( Loss_delta_w2( linspace(0.01,0.99,100) , zeros(1,100) ) );
% plot( [grad_Target1 ;grad_Target0]' );
% 
% legend({'target class=1','target class=0'}); title('Prediction*3 | Target==True')
% xlabel('Predicted Probability of Misclassification'); xticklabels(string([get(gca,'Xtick')]/100)); 
% ylabel('Loss'); ylim([0 , 5*median( [(grad_Target1),(grad_Target0)] ) ])
% 
% 
% % plot confusion matrices next to training losses
% loadNNetPerformance('compareresults')


%% INSPECT SIGNAL AT OPEN AND CLOSED TRANSITION 
% % [coeff,score,latent] =pca(squeeze(Xts2));
% % figure;subplot(2,1,1); plot(Xts2(:,1,1)); yyaxis right; bar(Ts_Y(:,1,2),'LineStyle','none','FaceAlpha',0.2); ylabel('Closed eyes=0')
% % subplot(2,1,2); plot(score(:,:)); yyaxis right; bar(Ts_Y(:,1,2),'LineStyle','none','FaceAlpha',0.2); ylabel('Closed eyes=0')
% 
% % see Boxplots of EEG transition data for EEG #1
% figure; 
% subplot(2,1,1);
% boxplot(reshape(eyeOseq(:,:,:),[numel(eyeOid)*numel(vars), (2*TI)+1,1]))
% ylim([0 1])
% subplot(2,1,2);
% boxplot(reshape(eyeCseq(:,:,:),[numel(eyeCid)*numel(vars), (2*TI)+1,1]))
% ylim([0 1])

% 
% % plot total variance in state transition, for all variables.
% EEGtransition_var=[];
% for V = 1:numel(vars)
%     EEGtransition_var(V, :,1 ) = std(eyeOseq(:,:,V),0,1,'omitnan' );
%     EEGtransition_var(V, :,2 ) = std(eyeCseq(:,:,V),0,1,'omitnan' );
% end
% 
% figure;
% for V=1:numel(vars)
%     subplot(5,1,V)
%     bar( squeeze( EEGtransition_var(V,:,:) ) ); legend('Opening','Closing')
% end

% 
% Torig = size(Ts_X,1);
% minibatchsize = floor((Torig)/(Nbatches*seqlen_in));
% lostTseq = rem(Torig,Nbatches*seqlen_in);
% T = Torig-lostTseq;
% index = reshape( [1:T] , [Nbatches*seqlen_in , minibatchsize] )';
% Nfeatures=size(Ts_X,3);
% batchindices = cell(1,Nbatches);
% for bi=1:Nbatches
%     batchindices{bi} = index(:,((bi-1)*seqlen_in)+1:seqlen_in*bi);
% %     DataX{bi} = reshape( Ts_X( batchindices{bi} ,1,:) , [minibatchsize, seqlen_in, Nfeatures] );
% %     Yind=seqlen_in + batchindices{bi};
% %     DataY{bi} = reshape( Ts_Y( Yind(1:end-1,1:seqlen_out) ,1,:) , [minibatchsize-1, seqlen_out, Nfeatures] );
% %     DataY{bi} = reshape( Ts_Y( Yind(1:end,1:seqlen_out) ,1,:) , [minibatchsize-1, seqlen_out, Nfeatures] );
% end
% [DataXalt, DataYalt] = deal(cell(1,Nbatches));
% [Yind,BIAll]=deal(cell(Nbatches,1));
% for bi=1:Nbatches
%     BI=[];
%     for rw=1:size(batchindices{bi},1)
%         for ti=1:seqlen_in-1
%         BI(ti+1,:) = batchindices{bi}(rw,:)+ti;
%         end
%         BI(1,:) = batchindices{bi}(rw,:);
%         BIAll{bi} = [BIAll{bi};BI];
%     end
%     DataXalt{bi} = reshape( Ts_X( BIAll{bi} ,1,:) , [minibatchsize*seqlen_in, seqlen_in, size(Ts_X,3)] );
%     Yind{bi}=seqlen_in + BIAll{bi};
% %     DataYalt{bi} = reshape( Ts_Y( Yind(1:end-1,1:seqlen_out) ,1,:) , [minibatchsize*seqlen_in-1, seqlen_out, 1] );
%     DataYalt{bi} = reshape( Ts_Y( Yind{bi}(1:end,1:seqlen_out) ,1,:) , [minibatchsize*seqlen_in, seqlen_out, size(Ts_Y,3)] );
% end
% Yind_c = cat(1,Yind{:});
% BIAll_c = cat(1,BIAll{:});rw=1

%% Check if data is aggregated correctly.
% figure; [XX,YY]=deal( NaN(singleseqlen,batchsize*Nbatches));
% for is = 1:singleseqlen
%     rwini=20;
%     rw=rwini + seqlen_in*(is-1)
%     X=[];%NaN(1,(rw-1)*(is==1));
%     for bi = 1:Nbatches
%         X = [X,DataX_batched{bi}(rw,:,1)]; 
%     end
%     X = [NaN( 1, (Nbatches*seqlen_in)*(is-1)),X];
%     XX(is,rwini-1+[(Nbatches*seqlen_in)*(is-1)+1 : numel(X)]) = X([(Nbatches*seqlen_in)*(is-1)+1 : numel(X)]);
% 
%     Y=NaN(1,(rw-1)*(is==1));
%     for bi = 1:Nbatches
% %         Y = [Y,DataY_batched{bi}(rw,1,1)]; 
%         Y = [Y,NaN(1,seqlen_in),DataY_batched{bi}(rw,1,2)]; 
%     end
%     Y = [NaN( 1, (Nbatches*seqlen_in)*(is-1)),Y];
%     YY(is,(rw-1)*(is==1)+ [(Nbatches*seqlen_in)*(is-1)+1: numel(Y)] ) =Y( (Nbatches*seqlen_in)*(is-1)+1: numel(Y) );
% end
% %     yyaxis left; hold on;
%     plot(1:size(XX,2),XX)
%     yyaxis right; bar( nanmean(YY,1),'LineStyle','none'); hold on;
%     yyaxis right; bar(Ts_Y(:,1,2),'LineStyle','none','FaceAlpha',0.2); ylabel('Closed eyes=0')
% 
% yyaxis left; hold on;
% rw=10; X=[];for bi = 1:Nbatches, X = [X,DataXalt{bi}(rw,:,1)]; end
% plot([NaN(1,seqlen_in*bi + (rw-seqlen_in)),X])
% rw=10; Y=[];for bi = 1:Nbatches, Y = [Y,NaN(1,seqlen_in),DataYalt{bi}(rw,:,1)]; end
% yyaxis right; bar([NaN(1,seqlen_in*bi + (rw-seqlen_in)),Y],'LineStyle','none','FaceAlpha',0.2);

   
    
    function [GDOptimizer, learnrate, epochs, NNModels, LossType] = test_1(InputDataSize,OutputDataSize)
    epochs = 850;
    GDOptimizer = 'Adam';
    learnrate = 0.005;
    LossType = "BinaryCrossEntropy";
    %% Topology
    % input embedding
    encoderHUs = 30;
    decoderHUs = 30;
    classifierHUs = OutputDataSize(3);
    
    % create Encoder
    [NNLayerEnc] = GenerateNNetLayer( encoderHUs , InputDataSize(1) , InputDataSize(3) , "LSTM" , "tanh" ...
        , InputDataSize(2) , struct('resetstate',false,'predictsequence',false ) );
    % create Decoder
    [NNLayerDec] = GenerateNNetLayer( decoderHUs , InputDataSize(1) , NNLayerEnc.Nunits , "LSTM" , "tanh" ...
        , OutputDataSize(2) , struct('resetstate',false,'predictsequence',true ) );
    
    % Projection Layer to Output tokens
    [NNLayerFinal] = GenerateNNetLayer( classifierHUs , OutputDataSize(1) , NNLayerDec.Nunits , "dense" , "softmax" );
    
    NNModels = [{NNLayerEnc},{NNLayerDec},{NNLayerFinal}];
    end

