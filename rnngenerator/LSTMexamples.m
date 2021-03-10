%% EXAMPLE USE OF THE SEQUENTIAL LSTM BUILDER / TRAINER
% BINARY CALSSIFICATION TASK, LAGGED SINE WAVE WITH 2 TYPES OF NOISE.
% ENCODER DECODER SETUP


seqlen_in = 6; % input sequence length
seqlen_out = 3; % target sequence length
nclasses_out = 2; % number of classes in Y. binary
Nbatches = 4; % number of batches in each training epoch
obsperatch = 200;

observations = obsperatch.*Nbatches;


% two class categorical output in 3rd dimension)
pureSignal = sin(linspace(1,20*pi,observations));
% output is a 1 in the first class if pure sine signal > 0, and a zero in
% the second class.
Ts_Y = permute( cat(3, double(pureSignal>=0), double(pureSignal <0) ),[2 1 3]);
% input signal is 3 dimensions of noisy version of 1st dimension of puresignal, followed by equal length sequence of noise
Ts_X = permute( cat(3, pureSignal.*(1+ rand(1,observations)*0.5) , pureSignal.*(1 -rand(1,observations)*0.5) ) , [2 1 3] );

% create minibatches of the sequences of variables.
[DataX_batched, DataY_batched, Xind_c, Yind_c, singleseqlen] = batchdatasets( Nbatches, Ts_X , Ts_Y , 'seq2batch' ,seqlen_in, seqlen_out);

% size of each minibatch input and output
InputDataSize = size(DataX_batched{1});
OutputDataSize = size(DataY_batched{1});
batchsize = size(DataX_batched{1},1);


%% Topology
    
    % input embedding
    encoderHUs = 20;
    decoderHUs = 20;
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

%% Training settings

    epochs = 50;
    
    trainingSettings = struct();
    trainingSettings.LossType = "MultiClassCrossEntropy"; % binary classification done with two one-hot coded output vectors
    trainingSettings.learnrate = 0.005;
    trainingSettings.GDOptimizer = 'Adam';
    trainingSettings.gradclip = true;

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