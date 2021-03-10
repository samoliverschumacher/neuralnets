function [X, Y, varargout] = batchdatasets( batchsize , DataX, DataY, varargin)

if nargin>3
    switch varargin{1}
        case 'unbatch_seq2batch'
              % unroll prediction, take the average of each sequence from consecutive
              % starting points
              Xind_c = varargin{2};
              Yind_c = varargin{3};
              
              DataY_unbatched = cat(1,DataY{:});
              DataX_unbatched = cat(1,DataX{:});
              
              X = NaN( max(Xind_c(:)) , size(DataX_unbatched,3) );
              Y = NaN( max(Yind_c(:)) , size(DataY_unbatched,3) );
              
              % recreate entire Target sequence from unbatched data
              for fy = 1:size(Y,2)
                    Y_temp = reshape(DataY_unbatched(:,:,fy), [numel(DataY_unbatched(:,:,fy)) , 1] );
                    Y(:,fy) = accumarray( Yind_c(:) , Y_temp ,[],@mean , NaN);
              end
              
              for fx = 1:size(X,2)
                    X_temp = reshape(DataX_unbatched(:,:,fx), [numel(DataX_unbatched(:,:,fx)) , 1] );
                    X(:,fx) = accumarray( Xind_c(:) , X_temp ,[],@mean , NaN);
              end
                    
                           
              
              
              
        case 'seq2batch'
            % When the input X data is a long tiemseries dataset, which
            % needs to be split such that each batch is a continuation of
            % observations in the prior batch. batch1, obs1 -> batch2, obs1
            % INPUT data:
            % DataX = [Timesteps x scalar Variables]
            % DataY = [Timesteps x scalar Output]
            
            seqlen_in = varargin{2};
            seqlen_out = varargin{3};
            
            
            Nbatches = batchsize;
            
            
            Torig = size(DataX,1);
            
            singleseqlen = floor((Torig)/(Nbatches*seqlen_in));
            
            lostTseq = rem(Torig,Nbatches*seqlen_in);
            
            T = Torig-lostTseq;
            
            index = reshape( [1:T] , [Nbatches*seqlen_in , singleseqlen] )';
            
            Nfeatures=size(DataX,3);
            
            batchindices = cell(1,Nbatches);
            for bi=1:Nbatches
                batchindices{bi} = index(:,((bi-1)*seqlen_in)+1:seqlen_in*bi);
                %     DataX{bi} = reshape( Ts_X( batchindices{bi} ,1,:) , [minibatchsize, seqlen_in, Nfeatures] );
                %     Yind=seqlen_in + batchindices{bi};
                %     DataY{bi} = reshape( Ts_Y( Yind(1:end-1,1:seqlen_out) ,1,:) , [minibatchsize-1, seqlen_out, Nfeatures] );
                %     DataY{bi} = reshape( Ts_Y( Yind(1:end,1:seqlen_out) ,1,:) , [minibatchsize-1, seqlen_out, Nfeatures] );
            end
            
            [DataXalt, DataYalt] = deal(cell(1,Nbatches));
            [Yind,Xind]=deal(cell(Nbatches,1));
            for bi=1:Nbatches
                BI=[];
                for rw=1:size(batchindices{bi},1)
                    for ti=1:seqlen_in-1
                        BI(ti+1,:) = batchindices{bi}(rw,:)+ti;
                    end
                    BI(1,:) = batchindices{bi}(rw,:);
                    
                    Xind{bi} = [Xind{bi};BI];
                end
                DataXalt{bi} = reshape( DataX( Xind{bi} ,1,:) , [singleseqlen*seqlen_in, seqlen_in, size(DataX,3)] );
                Yind{bi}=seqlen_in + Xind{bi}( :, 1:seqlen_out );
                %     DataYalt{bi} = reshape( Ts_Y( Yind(1:end-1,1:seqlen_out) ,1,:) , [minibatchsize*seqlen_in-1, seqlen_out, 1] );
                DataYalt{bi} = reshape( DataY( Yind{bi}(1:end,1:seqlen_out) ,1,:) , [singleseqlen*seqlen_in, seqlen_out, size(DataY,3)] );
            end
            Yind_c = cat(1,Yind{:});
            Xind_c = cat(1,Xind{:});

            Y = DataYalt';
            X = DataXalt';
            varargout{1} = Xind_c;
            varargout{2} = Yind_c;
            varargout{3} = singleseqlen;
            
        otherwise
    end
else
    % INPUTS: [ Observations x timesteps x features ]
    removeLastObs = rem( size(DataX,1) ,batchsize);
    if removeLastObs~=0
        sprintf(' REMAINDER OBSERVATIONS LOST DUE TO BATCH SIZING: %0.0f',removeLastObs)
        %     disp( DataX( end-(removeLastObs-1) , :) )
        
    end
    DataX = DataX(1:end-(removeLastObs),:,: , : );
    DataY = DataY(1:end-(removeLastObs),:,: , : );
    
    
    if ~iscell(DataX)
        DataX = arrayfun(@(K) DataX(K,:,: , : ) , [1:size(DataX,1)]' ,'un', 0);
        DataY = arrayfun(@(K) DataY(K,:,: , : ) , [1:size(DataY,1)]' ,'un', 0);
    end
    
    d1 = repmat( batchsize , [ceil(size(DataX,1)/batchsize)  , 1] );
    d1(end) = size(DataX,1) - sum(d1(1:end-1));
    X = mat2cell( cat(1,DataX{:,1}) , d1 ); %
    X(cellfun(@(K) size(K,1) , X)~=batchsize,:) = [];
    
    d2 = repmat( batchsize , [ceil(size(DataY,1)/batchsize)  , 1] );
    d2(end) = size(DataY,1) - sum(d2(1:end-1));
    Y = mat2cell( cat(1,DataY{:,1}) , d2 );
    Y(cellfun(@(K) size(K,1) , Y)~=batchsize,:) = [];
end

end