function [fighndl, fighndl2] = plotPrediction( yPred_binom , yTarget_binom, yPred_prob , yTarget_prob , steplength_seconds , varargin )

%%%% 1 IS FOR EYES CLOSED, 0 IS EYES OPEN

if nargin>5
    plottype = varargin{1};
else
    plottype ='';
end
if contains(plottype,'timeseries') || numel(plottype)==0
    fighndl = figure('Position',[141 471 1471 470]);
    plot(yPred_prob)
    yyaxis right; bar(yTarget_binom,'LineStyle','none','FaceAlpha',0.2);
    xlim([1,numel(yPred_prob)]); set(gca,'Ytick',[0 1])
    yyaxis left;
    ylabel('Prob( Eyes Closed )');
    correctIdx = (yPred_binom == yTarget_binom);
    hold on; plot( find(correctIdx) , yTarget_binom(correctIdx) ,'d','MarkerFaceColor','g','Color','none')
    hold on; plot( find(~correctIdx) , yTarget_binom(~correctIdx) ,'d','MarkerFaceColor','r','MarkerSize',2)
    legend({'Average Predicted Eye State','Correct Prediction','Incorrect Prediction','Average Target Eye State'},'Location','north')
    title(['Average State across windows of length ',num2str(steplength_seconds),' seconds'])
    xlabel('Timestep');
    
end
if contains(plottype,'confusion') || numel(plottype)==0
    figure;
    subplot(4,1,1)
    plotconfusion(categorical(yTarget_binom),categorical(yPred_binom) )
    fighndl2 = gcf;
    set(fighndl2,'Units','normalized','Position',[(2/5) (2/5) 0.2 0.3]); 
%     set(fighndl2,'Position',[758,462,261,222])
%     set(fighndl2,'InnerPosition',[758,462,261,222])
    set(get(gca,'Title'),'FontSize',9);set(get(gca,'Title'),'FontWeight','normal');
    set(get(gca,'Xlabel'),'FontSize',7)
    set(get(gca,'Ylabel'),'FontSize',7)
    set(fighndl2,'Units','normalized','Position',[(2/5) (2/5) 0.2 0.3]); 
    set(fighndl2,'PaperPosition',[1 1 1 1]); 
    
    width=6; %width of figure in cm
    height=6; %height of figure in cm
    set(fighndl2,'units','centimeters','position',[0 0 width height])
end

if ~exist('fighndl2','var')
    fighndl2 =[];
end
if ~exist('fighndl','var')
    fighndl =[];
end
end
% 
% function [fgnew,AXnew] = MovePlots(SourcePlotFig_1, SourcePlotFig_2)
% 
% % First, create 4  figures with four different graphs (each with  a 
% % colorbar):
% figure(1) % SourcePlotFig(1) = figure(1);
% surf(peaks(4))
% colorbar
% figure(2) %SourcePlotFig(2)=figure(2);
% mesh(peaks(4))
% colorbar
% 
% 
% % Now create destination graph
% figure(3) % DestPlotFig = figure(3);
% ax = zeros(1,2);
% ax(1) = subplot(1,4,1);
% ax(2) = subplot(1,4,2:4);
% 
% % Now copy contents of each figure over to destination figure
% % Modify position of each axes as it is transferred
% for ii = [1 , 2] % cycle through existing source figures
%       figure(ii)
%       h = get(gcf,'Children'); %h = get(gcf,'Children');
%       newh = copyobj(h,3);%copyobj(h,DestPlotFig);
%       for j = 1:length(newh)
%             posnewh = get(newh(j),'Position');
%             possub  = get(ax(ii),'Position');
%             set(newh(j),'Position',...
%                   [possub(1) posnewh(2) possub(3) posnewh(4)])% [posnewh(1) possub(2) posnewh(3) possub(4)])
%       end
%       delete(ax(ii));
% end
% figure(3)%; DestPlotFig
% 
% 
% % First, create 4  figures with four different graphs (each with  a 
% % colorbar):
% figure(1)
% surf(peaks(10))
% colorbar
% figure(2)
% mesh(peaks(10))
% colorbar
% 
% % Now create destination graph
% figure(3)
% % ax = zeros(1,2);
% ax(1) = subplot(1,4,1);
% ax(2) = subplot(1,4,2:4);
% % for i = 1:4
% %     ax(i)=subplot(4,1,i);
% % end
% % Now copy contents of each figure over to destination figure
% % Modify position of each axes as it is transferredj = 2
% for i = 1:2
%       figure(i)
%       h = get(gcf,'Children');
%       newh = copyobj(h,3);
%       for j = 1:length(newh)
%             posnewh = get(newh(j),'Position');
%             possub  = get(ax(i),'Position');
%             newX = posnewh(1);
%             newY = possub(2);
%             newWid =  posnewh(3);
%             newH = possub(4);
%             set(newh(j),'Position',...
%                   [newX newY newWid newH])
%       end
%       delete(ax(i));
% end
% figure(3)
% % 
% % 
% % First, create 4  figures with four different graphs (each with  a 
% % colorbar):
% figure(1)
% scatter(rand(10,1),ones(10,1)) %surf(peaks(10))
% colorbar;
% title('Num1');
% yyaxis right; scatter(rand(10,1)*5,ones(10,1)) %surf(peaks(10))
% figure(2)
% mesh(peaks(10))
% colorbar
% figure(3)
% contour(peaks(10))
% colorbar
% figure(4)
% pcolor(peaks(10))
% colorbar
% % Now create destination graph
% figure(5)
% % ax = zeros(4,1);
% for i = 1:4
%       ax(i)=subplot(4,1,i);
% end
% % Now copy contents of each figure over to destination figure
% % Modify position of each axes as it is transferred
% for i = 1:4
%       figure(i)
%       h = get(gcf,'Children');
%       newh = copyobj(h,5)
%       for j = 1:length(newh)
%             posnewh = get(newh(j),'Position');
%             possub  = get(ax(i),'Position');
%             newX = posnewh(1);
%             newY = possub(2);
%             newWid =  posnewh(3);
%             newH = possub(4);
%             set(newh(j),'Position',...
%                   [newX newY newWid newH])
%       end
%       delete(ax(i));
% end
% figure(5)

% close all