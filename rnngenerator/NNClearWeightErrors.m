function [NNLayer] = NNClearWeightErrors( NNLayer )

switch NNLayer.Type
    case "LSTM"
        
        for ffn = fieldnames(NNLayer.dEdW.wIN)'
            NNLayer.dEdW.wIN.(ffn{1}) = zeros( size(NNLayer.dEdW.wIN.(ffn{1})) );
            NNLayer.dEdW.wrec.(ffn{1}) = zeros( size(NNLayer.dEdW.wrec.(ffn{1})) );
            NNLayer.dEdW.wb.(ffn{1}) = zeros( size(NNLayer.dEdW.wb.(ffn{1})) );
        end
        
        if NNLayer.Peephole
            for ffn = fieldnames(NNLayer.dEdW.wpeep)'
                NNLayer.dEdW.wpeep.(ffn{1}) = zeros( size(NNLayer.dEdW.wpeep.(ffn{1})) );
            end
        end

    case "dense"
        
            NNLayer.dEdW.wIN = zeros( size(NNLayer.dEdW.wIN) );
            NNLayer.dEdW.wb = zeros( size(NNLayer.dEdW.wb) );
            
    case "general"
        
        NNLayer.dEdW = zeros( size(NNLayer.dEdW) );

    otherwise
        disp('thats not a type of NN layer....')
        
end