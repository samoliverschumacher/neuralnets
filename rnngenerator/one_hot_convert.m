function onehot = one_hot_convert( X , type , Nfeats , minX , maxX)
buckets = linspace(minX,maxX,Nfeats); %[minX:(maxX-minX)/Nfeats:maxX];
nsamples = size(X,1);
nvars = size(X,2);
Sz = [nsamples , nvars , Nfeats];
switch type
    case 'encode'
        onehot = zeros( Sz );
        
        for cc=1:nvars
            indx = discretize(X(:,cc), [buckets,buckets(end)+1] );
%             [~,indx ] = min(abs(X(:,cc)  - buckets));
            ohid = sub2ind(Sz,[1:nsamples]',cc.*ones(nsamples,1),indx);
            onehot(ohid) = 1;
        end
    case 'decode'
        if all(arrayfun(@(rw) nnz(X(rw,1,:))~=1 , 1:nsamples ))
            tmp = arrayfun(@(bt) arrayfun(@(cc) find(X( bt , cc ,:) == max( X( bt , cc ,:) ),1) , 1:nvars ) ,[1:size(X,1)] ,'un',0);
            onehot = buckets( cat(1,tmp{:})  ) ;
            
        else
            onehot = zeros( Sz(1:2) );
            for cc=1:nvars
                indx = arrayfun(@(rw) find(X( rw,cc,:)) , [1:nsamples]');
                onehot( : ,cc) = buckets(indx);
            end
        end
end


