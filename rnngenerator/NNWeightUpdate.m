function [NNLayer] = NNWeightUpdate( NNLayer , learnrate , GDOptimizer , OptParams , varargin )

batchaccum = @(X) sum( X , 3);

if nargin>4
    gradclipping = varargin{1};
    if gradclipping==true
        batchaccum = @(X) mean( X , 3);
    end
else
    gradclipping = false;
end

switch NNLayer.Type
    case "LSTM"
        % Update the weights with the summation of backpropogated error over time
        
        [NNLayer.Weights.wIN.forget , NNLayer.BP_pOut.wIN.forget] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wIN.forget ) , NNLayer.Weights.wIN.forget , learnrate , [ OptParams , NNLayer.BP_pOut.wIN.forget ] );
        [NNLayer.Weights.wIN.input , NNLayer.BP_pOut.wIN.input] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wIN.input ) , NNLayer.Weights.wIN.input, learnrate , [ OptParams , NNLayer.BP_pOut.wIN.input ] );
        [NNLayer.Weights.wIN.activate , NNLayer.BP_pOut.wIN.activate] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wIN.activate ) , NNLayer.Weights.wIN.activate, learnrate , [ OptParams , NNLayer.BP_pOut.wIN.activate ] );
        [NNLayer.Weights.wIN.output , NNLayer.BP_pOut.wIN.output] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wIN.output ) , NNLayer.Weights.wIN.output, learnrate , [ OptParams , NNLayer.BP_pOut.wIN.output ] );
        
        [NNLayer.Weights.wrec.forget , NNLayer.BP_pOut.wrec.forget] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wrec.forget ) , NNLayer.Weights.wrec.forget, learnrate , [ OptParams , NNLayer.BP_pOut.wrec.forget ] );
        [NNLayer.Weights.wrec.input , NNLayer.BP_pOut.wrec.input] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wrec.input ) , NNLayer.Weights.wrec.input, learnrate , [ OptParams , NNLayer.BP_pOut.wrec.input ] );
        [NNLayer.Weights.wrec.activate , NNLayer.BP_pOut.wrec.activate] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wrec.activate ) , NNLayer.Weights.wrec.activate, learnrate , [ OptParams , NNLayer.BP_pOut.wrec.activate ] );
        [NNLayer.Weights.wrec.output , NNLayer.BP_pOut.wrec.output] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wrec.output ) , NNLayer.Weights.wrec.output, learnrate , [ OptParams , NNLayer.BP_pOut.wrec.output ] );
        
        [NNLayer.Weights.wb.forget , NNLayer.BP_pOut.wb.forget] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wb.forget ) , NNLayer.Weights.wb.forget, learnrate , [ OptParams , NNLayer.BP_pOut.wb.forget ] );
        [NNLayer.Weights.wb.input , NNLayer.BP_pOut.wb.input] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wb.input ) , NNLayer.Weights.wb.input, learnrate , [ OptParams , NNLayer.BP_pOut.wb.input ] );
        [NNLayer.Weights.wb.activate , NNLayer.BP_pOut.wb.activate] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wb.activate ) , NNLayer.Weights.wb.activate, learnrate , [ OptParams , NNLayer.BP_pOut.wb.activate ] );
        [NNLayer.Weights.wb.output , NNLayer.BP_pOut.wb.output] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wb.output ) , NNLayer.Weights.wb.output, learnrate , [ OptParams , NNLayer.BP_pOut.wb.output ] );
        
        [NNLayer.Weights.wpeep.forget , NNLayer.BP_pOut.wpeep.forget] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wpeep.forget ) , NNLayer.Weights.wpeep.forget, learnrate , [ OptParams , NNLayer.BP_pOut.wpeep.forget ] );
        [NNLayer.Weights.wpeep.input , NNLayer.BP_pOut.wpeep.input] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wpeep.input ) , NNLayer.Weights.wpeep.input, learnrate , [ OptParams , NNLayer.BP_pOut.wpeep.input ] );
        [NNLayer.Weights.wpeep.output , NNLayer.BP_pOut.wpeep.output] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wpeep.output ) , NNLayer.Weights.wpeep.output, learnrate , [ OptParams , NNLayer.BP_pOut.wpeep.output ] );
        
% 	case "BiLSTM"
%         
%         [NNLayer.Forward] = NNWeightUpdate( NNLayer.Forward , learnrate , GDOptimizer , OptParams );
%         [NNLayer.Backward] = NNWeightUpdate( NNLayer.Backward , learnrate , GDOptimizer , OptParams );
            
    case "dense"
        %         batchaccum =@(X) sum( X , 3);
        [NNLayer.Weights.wIN , NNLayer.BP_pOut.wIN] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wIN ) , NNLayer.Weights.wIN , learnrate , [ OptParams , NNLayer.BP_pOut.wIN ] );
        [NNLayer.Weights.wb , NNLayer.BP_pOut.wb] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW.wb ) , NNLayer.Weights.wb, learnrate , [ OptParams , NNLayer.BP_pOut.wb ] );
        
    case "general"
        
        [NNLayer.Weights , NNLayer.BP_pOut] = NNGDescentOptimizer( GDOptimizer , batchaccum( NNLayer.dEdW ) , NNLayer.Weights , learnrate , [ OptParams , NNLayer.BP_pOut ] );
        
    otherwise
        disp('thats not a type of NN layer....')
        
end



    function [NewWeights , varargout] = NNGDescentOptimizer( GDOptimizer , dEdw , OldWeights, learnrate , Params )
        %%% Gradient Descent Optimizer
            
        if gradclipping
            gradthresh = 5;
            
            overthresh = (dEdw) > (gradthresh);
            underthresh = (dEdw) < (-gradthresh);
            dEdw(overthresh) = gradthresh;
            dEdw(underthresh) = -gradthresh;
            
        end
        
        switch GDOptimizer
            case 'Vanilla'
                if isempty(Params{1})
%                     delweights = - learnrate .* (dEdw);
                    NewWeights = OldWeights - learnrate .* (dEdw);
                else
%                     wdecayL = Params{1};
%                     currentiteration = Params{2};
%                     delweights = - (learnrate .* max( exp(-wdecayL.*currentiteration) , 0.15 )) .* dEdw;
%                     NewWeights = OldWeights - (learnrate .* max( exp(-wdecayL.*currentiteration) , 0.15 )) .* dEdw;
                    NewWeights = OldWeights - (learnrate .* max( exp(-Params{1}.*Params{2}) , 0.15 )) .* dEdw;
                end
%                 NewWeights = OldWeights + delweights;
                varargout = cell(nargout,1);
                
            case 'LRschedule'
                %                 schedFunc = @(it , epochs, minlr, maxlr) minlr + maxlr*(it / (epochs/2)) - 4.*(it>(epochs/2)).*(maxlr)*( (it - (epochs/2))/epochs );
                schedFunc = Params{1}{1};
                lr = schedFunc( Params{2} , Params{1}{2} , Params{1}{3} , Params{1}{4} );
                delweights = - lr .* (dEdw);
                NewWeights = OldWeights + delweights;
                
            case 'Momentum'
                mm_gamma = Params{1};
                delW_last = Params{2};
                delweights = - learnrate.*dEdw - (mm_gamma.*delW_last);
                NewWeights = OldWeights + delweights;
                
                varargout{1}  = delweights; % store the change in weights for next iterations update
            case 'Adam'
                decayrate1 = Params{1}(1); % beta 1
                decayrate2 = Params{1}(2); % beta 2
                t = Params{2};
                Momentum = Params{3};
                %                 m_prior = Params{3}(:,:,1); % last first-moment
                %                 v_prior = Params{3}(:,:,2); % last second-moment
                epssmooth = 1e-7;
                
                m1 = decayrate1.*Momentum(:,:,1) + (1-decayrate1).*dEdw; % m1 = decayrate1.*m_prior + (1-decayrate1).*dEdw;
                v1 = decayrate2.*Momentum(:,:,2) + (1-decayrate2).*(dEdw.^2); % v1 = decayrate2.*v_prior + (1-decayrate2).*(dEdw.^2);
                % Option 1
%                 mbar = m1./(1-decayrate1);
%                 vbar = v1./(1-decayrate2);
%                 
%                 delweights = - learnrate .* ( mbar ./ (sqrt(vbar) + epssmooth));
                % Option 2
                mbar = m1./(1-(decayrate1.^t));
                vbar = v1./(1-(decayrate2.^t));
                
                delweights = - learnrate .* ( mbar ./ (sqrt(vbar) + epssmooth));
                
                
                NewWeights = OldWeights + delweights;
                
                Momentum(:,:,1) = m1;
                Momentum(:,:,2) = v1;
                varargout{1}  = Momentum;
                %                 varargout{1}  = cat(3,m1,v1);% [{m1} , {v1}];
                
            case 'Adagrad'
                SumSqsPastGradients = Params{1};
                varargout{1}  = SumSqsPastGradients; % store the change in weights for next iterations update
                
            case 'Adadelta' % https://arxiv.org/pdf/1212.5701.pdf
                %         oldDecaingAvgSqdGradients = Params{1};
                %         newDecaingAvgSqdGradients = ad_gamma.*oldDecaingAvgSqdGradients + (1-ad_gamma).*dEdw;
                %         epssmooth = 1e-8; %prevents division by Zero
                %         RMSgradient = sqrt( newDecaingAvgSqdGradients + epssmooth );
                %         delweights = - ( learnrate./ RMSgradient ).*dEdw;
                %         NewWeights = OldWeights + delweights;
                %  -   -   -   -   -   %  -   -   -   -   -   %  -   -   -   -   -
                ad_gamma = Params{1}; % try 0.9
                OldSqrdAvgdelweights = Params{3}(:,:,1);
                OldSqrdAvgGradients = Params{3}(:,:,2);
                epssmooth = 1e-12; %prevents division by Zero
                
                NewSqrdAvgGradients = ad_gamma.*OldSqrdAvgGradients + (1-ad_gamma).*(dEdw.^2);
                RMSgradients = sqrt( NewSqrdAvgGradients + epssmooth ); % (g_t)
                RMSOlddelChange = sqrt( OldSqrdAvgdelweights + epssmooth );
                delweights = - (RMSOlddelChange ./ RMSgradients).*dEdw; % -learnrate / sqrt( g_t^2 + epssmooth )
                NewWeights = OldWeights + delweights;
                
                NewSqrdAvgdelweights = ad_gamma.* OldSqrdAvgdelweights + (1-ad_gamma).*(delweights.^2);
                
                varargout{1}  = cat(3,NewSqrdAvgdelweights,NewSqrdAvgGradients);%[{NewSqrdAvgdelweights} , {NewSqrdAvgGradients}]; % store the change in weights for next iterations update
                
                            case 'HyperGradientDescent' % Medium Article: "Towards Faster Training and Samller Generalisation Gaps in Deep Learning 
                                
            otherwise
                disp('optimizer selected not available')
        end
        
        if any( isnan( NewWeights(:) ) )
            keyboard
        end
    end

end