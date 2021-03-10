function [plthandles] = PlotTraining( NNModels , it , plthandles , PredErr , PredErrorGrad )
DoPlots = true;
if DoPlots
firsttime = isempty(plthandles);
% get layer type
LayerTypes = arrayfun(@(x) x{1}.Type , NNModels);

% initialise plots
xnew = it;

if firsttime
    figure;
    
end


% Overall Prediction Error
subplot(2,2,1)
if firsttime
    plthandles = plot( xnew , PredErr );
else
    plthnd = plthandles(1);
    ycur = get(plthnd,'YData');
    xcur = get(plthnd,'XData');
    
    ynew = PredErr;
    
    set( plthnd , 'XData', [xcur , xnew] , 'YData' , [ycur , ynew] );
    drawnow;
end

subplot(2,2,2)
if firsttime
    plthandles(2) = plot( xnew , PredErrorGrad );
else
    plthnd = plthandles(2);
    ycur = get(plthnd,'YData');
    xcur = get(plthnd,'XData');
    
    ynew = PredErr;
    
    set( plthnd , 'XData', [xcur , xnew] , 'YData' , [ycur , ynew] );
    drawnow;
end

% Per Layer Averages for Input Weight gradients
avInGrad = zeros( numel(NNModels) , 4);
% Extract gradients from models
for lay = 1:numel(NNModels)
    switch LayerTypes(lay)
        case "LSTM"
        if ~NNModels{lay}.BiLSTM
            avInGrad( lay , 1 ) = (norm([NNModels{lay}.dEdW.wIN.forget(:);NNModels{lay}.dEdW.wIN.input(:);NNModels{lay}.dEdW.wIN.activate(:);NNModels{lay}.dEdW.wIN.output(:)])) ...
                ./ numel([NNModels{lay}.dEdW.wIN.forget(:);NNModels{lay}.dEdW.wIN.input(:);NNModels{lay}.dEdW.wIN.activate(:);NNModels{lay}.dEdW.wIN.output(:)]);
            if NNModels{lay}.Attention && NNModels{lay}.AttnInfo.ScoringFcn=="general"
                avInGrad( lay , 2 ) = (norm([NNModels{lay}.AttnInfo.dEdW(:)]))./numel(NNModels{lay}.AttnInfo.dEdW(:));
            end
            
            weightcount(lay) = [numel(NNModels{lay}.dEdW.wIN.forget).*4];
        else
        
            avInGrad( lay , 1 ) = norm( [NNModels{lay}.Forward.dEdW.wIN.forget(:);NNModels{lay}.Forward.dEdW.wIN.input(:);NNModels{lay}.Forward.dEdW.wIN.activate(:);NNModels{lay}.Forward.dEdW.wIN.output(:) ...
                ;NNModels{lay}.Backward.dEdW.wIN.forget(:);NNModels{lay}.Backward.dEdW.wIN.input(:);NNModels{lay}.Backward.dEdW.wIN.activate(:);NNModels{lay}.Backward.dEdW.wIN.output(:) ] ) ...
                ./ numel([NNModels{lay}.Forward.dEdW.wIN.forget(:);NNModels{lay}.Forward.dEdW.wIN.input(:);NNModels{lay}.Forward.dEdW.wIN.activate(:);NNModels{lay}.Forward.dEdW.wIN.output(:) ...
                ;NNModels{lay}.Backward.dEdW.wIN.forget(:);NNModels{lay}.Backward.dEdW.wIN.input(:);NNModels{lay}.Backward.dEdW.wIN.activate(:);NNModels{lay}.Backward.dEdW.wIN.output(:) ]);
            
            weightcount(lay) = [numel(NNModels{lay}.Forward.dEdW.wIN.forget).*4 + numel(NNModels{lay}.Backward.dEdW.wIN.forget).*4];
        end
        

        case "dense"
            avInGrad( lay , 1 ) = norm( (NNModels{lay}.dEdW.wIN(:)) ) / numel((NNModels{lay}.dEdW.wIN(:)));
            
            weightcount(lay) = [numel(NNModels{lay}.dEdW.wIN)];
    end
    
end
    

if 0==sum(avInGrad(:,2) )
else
    weightcount = [weightcount , numel(NNModels{end}.AttnInfo.dEdW)];
end
weightcount = weightcount./max(weightcount);

subplot(2,2,3)
ynew = avInGrad(:,:);
ynew(1,1)=ynew(1,1)./weightcount(1);
ynew(2,1)=ynew(2,1)./weightcount(2);
if 0==sum(avInGrad(:,2) )
else
    ynew(2,2)=ynew(2,2)./weightcount(3);
end
if firsttime
    plthandles(3:5) = plot( xnew, ynew(1,1) ,xnew, ynew(2,1)  ,xnew, ynew(2,2));
    legend({'lay 1','lay 2','lay Attn'})
    title('2norm dEdW')
else
    plthnd = plthandles(3);
    ycur = get(plthnd,'YData');
    xcur = get(plthnd,'XData');
    
    hold on;
    set( plthnd , 'XData', [xcur , xnew] , 'YData' , [ycur , ynew(1,1)] );
    
    plthnd = plthandles(4);
    ycur = get(plthnd,'YData');
    xcur = get(plthnd,'XData');
    
    set( plthnd , 'XData', [xcur , xnew] , 'YData' , [ycur , ynew(2,1)] );
    
    plthnd = plthandles(5);
    ycur = get(plthnd,'YData');
    xcur = get(plthnd,'XData');
    
    set( plthnd , 'XData', [xcur , xnew] , 'YData' , [ycur , ynew(2,2)] );
    hold off;
    
    drawnow;
end

else
    
end








