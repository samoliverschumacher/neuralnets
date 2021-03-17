function [Nbatches, OptParams, lrsched, classificationproblem,  LossFcn, Loss_delta, OutActiv , dCda, ErrorMetric1 , ErrorMetric2, Prediction, sampleidx, varargout] = initialiseNNet(NNModels,DataX_batched,DataY_batched, GDOptimizer, LossType , epochs, learnrate )
NNLayerFinal = NNModels{end};
gradnormalisation = true;
varargout{1} = gradnormalisation;
%% Dataset
Nobservations = size(cat(1,DataX_batched{:}),1);
Nbatches = size(DataX_batched,1);
batchsize = size(DataX_batched{1},1);
insize = [1 , size(DataX_batched{1} ,2) , size(DataX_batched{1} , 3)];%size(DataX_batched{1}(1,:,:)); % size of a single observation. always 1 row. column for features or timesteps, dim3 for features
outsize = [1 , size(DataY_batched{1} ,2) , size(DataY_batched{1} , 3)];%size(DataY_batched{1}(1,:,:)); % size of a single observation. always 1 row. column for features or timesteps, dim3 for features
if contains(NNLayerFinal.Type,'LSTM')
    Noutfeatures = outsize(3);
else
    Noutfeatures = outsize(3);%Noutfeatures = outsize(2);
end

%% Training Optimiser
switch GDOptimizer
    case 'Vanilla'
        wdecayL = 0*(0.0025).*(1000./epochs); % set to Zero for no decay
        OptParams = [{wdecayL} , {0}];% Params=[]; % cell containing all parmaeters relevant to the selected optimiser
        if isempty(learnrate)
            learnrate = 0.001;
        end
        x0 = learnrate;
        x1 = learnrate*10;
    case 'Adam'
        OptParams = {[0.9 , 0.999] , [0]};%
        if isempty(learnrate)
            learnrate = 0.001;
        end
        x0 = learnrate;
        x1 = learnrate;
    case 'Adadelta'
        OptParams = {0.95};
        if isempty(learnrate)
            learnrate = 1;
        end
        x0 = learnrate;
        x1 = learnrate;
end
lrsched = [linspace( x0,x1 ,floor(epochs/2)) , linspace(x1,x0 ,ceil(epochs/2))];      

%% Loss function

classificationproblem = strcmpi(NNLayerFinal.ActFcn,"softmax");
if NNLayerFinal.Type=="LSTM"
    if any(contains( fieldnames( NNLayerFinal ) , 'ProjectionLayer'))
        classificationproblem = strcmpi(NNLayerFinal.ProjectionLayer.ActFcn,"softmax");
    end
end


% LossType = "WeightedBinaryClassification";

if classificationproblem==true
      switch LossType
            % MultiCLass cross-entropy
            case "MultiClassCrossEntropy"
                  LossFcn = @(Pred,Real) -sum( Real.*log(Pred) , 3 ); %CrossEntLoss
%                   Loss_delta =  @(Pred,Real) Pred-Real; %CrossEntLoss_delta
%                   Loss_delta = @(Pred,Real) (-Real./Pred) + ( (1-Real)./(1-Pred) );%CrossEntLoss_delta
                  Loss_delta = @(Pred,Real) CrossEntropyDeriv_ifisnan( Real, Pred );%CrossEntLoss_delta
            case "WeightedTwoClassCrossEntropy"
                  % When two classes exist in target variable, both are 0 or 1
                  % The 1 in the second dimension is weighted heavier
                  LossFcn = @(Pred,Real) -sum( Real.*log(Pred) , 3 ); %CrossEntLoss
                  % weighting is on class #2
                  Loss_delta_w1 = @(Pred,Real) ...
                        (Real(:,:,2)==1).*((-Real./(Pred.^(1+sin([Pred.*pi])))) + ( (1-Real)./(1-(Pred.^(1+sin([Pred.*pi])))) )) ...
                        + (Real(:,:,2)==0).*((-Real./Pred) + ( (1-Real)./(1-Pred) ));

            case "BinaryCrossEntropy"
                  % Binary cross-entropy
                  LossFcn = @(Pred,Real) -Real.*log(Pred) -(1-Real).*log(1-Pred);
                  Loss_delta = @(Pred,Real) CrossEntropyDeriv_ifisnan( Real, Pred );%CrossEntLoss_delta
            case "WeightedBinaryClassification"
                  LossFcn = @(Pred,Real) -sum( Real.*log(Pred) , 3 ); %CrossEntLoss
                  
                  Loss_delta_w1 = @(Pred,Real) ...
                        (Real==1).*((-Real./(Pred.^(1+sin([Pred.*pi])))) + ( (1-Real)./(1-(Pred.^(1+sin([Pred.*pi])))) )) ...
                        + (Real==0).*((-Real./Pred) + ( (1-Real)./(1-Pred) ));
                  
                  w = [(2) , (1)]; % [positive class weight , negative class weight]
                  Loss_delta_w2 = @(Pred,Real) (Real==1).*((-Real./(Pred.*w(1))) + ( (1-Real)./(1-(Pred.*w(1))) )) ...
                        + (Real==0).*((-Real./Pred.*w(2)) + ( (1-Real)./(1-Pred.*w(2)) ));
                  Loss_delta_w3 = @(Pred,Real) (Real==1).*((-Real./(Pred.^2)) + ( (1-Real)./(1-(Pred.^2)) )) ...
                        + (Real==0).*((-Real./Pred) + ( (1-Real)./(1-Pred) ));
                  Loss_delta = Loss_delta_w1;
            otherwise
                  LossFcn = @(Pred,Real) -sum( Real.*log(Pred) , 3 ); %CrossEntLoss
                  Loss_delta = @(Pred,Real) CrossEntropyDeriv_ifisnan( Real, Pred );%CrossEntLoss_delta
      end
else
      switch LossType
            case "L2"
                  LossFcn = @(Pred,Real) sqrt( mean((Pred-Real).^2 , 2 ) ); %L2loss
                  Loss_delta = @(Pred,Real) Pred-Real; %L2loss_delta
            case "L1"
                  LossFcn = @(Pred,Real) abs(Pred-Real); %L1loss
                  Loss_delta = @(Pred,Real) sign(Pred-Real); %L2loss_delta
            otherwise
                  LossFcn = @(Pred,Real) sqrt( mean((Pred-Real).^2 , 2 ) ); %L2loss
                  Loss_delta = @(Pred,Real) Pred-Real; %L2loss_delta
      end
end


[OutActiv , dCda] = deal( NaN( [batchsize , outsize(2:end)] ) );
[ErrorMetric1 , ErrorMetric2] = deal(NaN( Nbatches , epochs , Noutfeatures ));
Prediction = NaN( [Nobservations , outsize(2:end)] ); % Samples x Output size
sampleidx = reshape(1:Nbatches.*batchsize,batchsize,Nbatches);

end
% % Examine loss gradient
% figure; plot(-Loss_delta( linspace(0.01,0.99,100) , ones(1,100) ))
% hold on; plot(Loss_delta( linspace(0.01,0.99,100) , zeros(1,100) ))
% legend({'target class=1','target class=0'})
% xlabel('Predicted Probability of class=1'); ylabel('Loss'); xticklabels(string([get(gca,'Xtick')]/100))