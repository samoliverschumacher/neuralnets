layer_check =3; 3;

% Assuming hidden states and memory states are supposed to show
% similarities between observations for any given timestep, the more
% distributed the PA loadings are, the more diverse the memory states are &
% therefore an easier job the NNet has to discern one observation from the
% other. If memory vectors are very similar, 1st PC will explain huge
% amount of the variety in them across observations, indiating the network
% is not transforming observation information in a way that makes each
% observation as distinct as it could be from one another.


% SavedStates{lay}.Inistates.Cell( sampleidx(:,btc) , 1 , : ) = NNModels{lay}.Inistates.Cell;
%             SavedStates{lay}.Inistates.Hidden( sampleidx(:,btc) , 1 , : ) = NNModels{lay}.Inistates.Hidden;
      figure;    
subplot(2,2,1); Title = {"InitialState Memory";"(Encoded Information)"};
X = squeeze(NNModels{layer_check}.Inistates.Cell);
Plt2Pcas( X , Title ,90 );

subplot(2,2,2); Title = {"InitialState Hidden";"(Encoded Information)"};
X = squeeze(NNModels{layer_check}.Inistates.Hidden);
Plt2Pcas( X , Title , 90 );
%%
      figure;    
subplot(2,2,1); Title = {"InitialState Memory";"(Encoded Information)"};
X = squeeze(SavedStates{3}.Inistates.Cell( : , 1 , : ));
Plt2Pcas( X , Title ,450 );

subplot(2,2,2); Title = {"InitialState Hidden";"(Encoded Information)"};
X = squeeze(SavedStates{3}.Inistates.Hidden( : , 1 , : ));
Plt2Pcas( X , Title , 450 );

subplot(2,2,3); Title = {"Attn Scores"};
X = reshape(AttnScores,[size(AttnScores,1),size(AttnScores,2)*size(AttnScores,3)]);
Plt2Pcas( X , Title ,size(AttnScores,1) );

hold on; Title = [Title;{"Attn Scores @ t=1"}];
X = (AttnScores(:,:,2));
Plt2Pcas( X , Title ,size(AttnScores,1) );

hold on; Title = [Title;{"Attn Scores @ t=end"}];
X = (AttnScores(:,:,end));
Plt2Pcas( X , Title ,size(AttnScores,1) );

subplot(2,2,4)
heatmap( squeeze(mean(AttnScores,1)) )

%%
figure;
subplot(2,2,1); Title = {"InitialState Memory";"(Encoded Information)"};
X = squeeze(NNModels{layer_check}.Inistates.Cell);
Plt2Pcas( X , Title );

subplot(2,2,2); Title ={"'Hidden In";"(Encoded Information)"};
X = squeeze( NNModels{layer_check}.Activations.HiddenIn(:,1,:) );
Plt2Pcas( X , Title );


subplot(2,2,2)
X = squeeze( NNModels{layer_check}.Activations.HiddenIn(:,1,:) );
[coeff,score,latent] = pca( X );
for ii=1:min([size(X,1),20])
scatter(score(ii,1), score(ii,2) );
text(score(ii,1), score(ii,2),num2str(ii))
hold on;
end
title({"'Hidden In";"(Encoded Information)"})

subplot(2,2,3)
% lasttstep = arrayfun(@(r) find(OutputMask{btc}(r,:),1) , [1:size(OutputMask{btc},1)]');
lasttstep=ones(batchsize,1).*NNModels{layer_check}.Nstates;
for ii=1:length(lasttstep)
    X(ii,:) = squeeze(NNModels{layer_check}.Activations.HiddenOut(ii,lasttstep(ii),:));
end
[coeff,score,latent] = pca( X );
for ii=1:min([size(X,1),20])
scatter(score(ii,1), score(ii,2) );
text(score(ii,1), score(ii,2),num2str(ii))
hold on;
end
title('Hidden Out at final timestep')

subplot(2,2,4)
% lasttstep = arrayfun(@(r) find(OutputMask{btc}(r,:),1) , [1:size(OutputMask{btc},1)]');
lasttstep=ones(batchsize,1).*NNModels{layer_check}.Nstates;
for ii=1:length(lasttstep)
    X(ii,:) = squeeze(NNModels{layer_check}.Activations.Memory(ii,lasttstep(ii),:));
end
[coeff,score,latent] = pca( X );
for ii=1:min([size(X,1),20])
scatter(score(ii,1), score(ii,2) );
text(score(ii,1), score(ii,2),num2str(ii))
hold on;
end
title('Memory Cell at final timestep')

% SavedStates{lay}.HiddenOut( sampleidx(:,btc) , : , : )
% SavedStates{lay}.Memory( sampleidx(:,btc) , : , : )

lasttstep=ones(batchsize,1).*NNModels{layer_check}.Nstates;
for ii=1:length(lasttstep)
    X(ii,:) = squeeze(NNModels{layer_check}.Activations.Memory(ii,lasttstep(ii),:));
end
[coeff,score,latent] = pca( X );
for ii=1:min([size(X,1),20])
scatter(score(ii,1), score(ii,2) );
text(score(ii,1), score(ii,2),num2str(ii))
hold on;
end
title('Attention Scores')



function Plt2Pcas( X , dispTitle, varargin)
if nargin>2
    numobs = varargin{1};
else
    numobs = min([size(X,1),20]);
end

[~,score,~] = pca( X );
if numobs<=20
    for ii=1:numobs
        scatter(score(ii,1), score(ii,2) );
        text(score(ii,1), score(ii,2),num2str(ii))
        hold on;
    end
else
     scatter(score(:,1), score(:,2) ,'.');
end
title(dispTitle)
end


