function [NeuronIni] = InitialiseLSTMNeuron( Ntimesteps, batchsize, Ini_PriorEpochThisbatch, Ini_LastBatch)
NeuronIni = Ini_PriorEpochThisbatch;

%% Make obsolete this function
return


%%


if Ntimesteps>1
    NeuronIni.Hidden(:,2:end,:) = [];
    NeuronIni.Cell(:,2:end,:) = [];
end

batchsequenceID = [1:Ntimesteps] + ([1:batchsize]'-1);
    idx1A = [1:min(batchsize,Ntimesteps)];
    
    idx0 = [1:min(batchsize,Ntimesteps)];
if batchsize>Ntimesteps % (Ntimesteps - batchsize) < 0 % store from prior epoch, or split batches further so hidden state can be calculated before initialising
    % subscripts in this batch for prior epoch    
    NeuronIni.Hidden( idx0 , 1 , :) = Ini_LastBatch.Hidden( batchsize , idx1A , :);
    NeuronIni.Cell( idx0 , 1 , :) = Ini_LastBatch.Cell( batchsize , idx1A , :);
    A = ( 1:batchsize-Ntimesteps );
    flpA = flip( A );
    for ii=A
%         [cl,rw] = find( batchsequenceID'==((Ntimesteps+batchsize-1)-ii-2) ,1,'last');
        [cl,rw] = find( batchsequenceID'==(flpA(ii)+Ntimesteps-1) ,1,'last'); % 'last' so that it doesnt repeatedly return the timestep which initially was zero-initialised. instead returns the timestep in each sample which had at least a few timesteps of prior reccurent activations built up as context, to create the state sampled used for initalising the activation of this current batches' timestep.
        NeuronIni.Hidden(flpA(ii)+Ntimesteps, 1 ,:) = Ini_PriorEpochThisbatch.Hidden( rw , cl );
        NeuronIni.Cell(flpA(ii)+Ntimesteps, 1 ,:) = Ini_PriorEpochThisbatch.Cell( rw , cl );
    end

else
    idx1 = [1:min(batchsize,Ntimesteps)]; %+ diff([batchsize,Ntimesteps]) ;
    NeuronIni.Hidden(idx0, 1 , :) = Ini_LastBatch.Hidden( batchsize , idx1 , :);
    NeuronIni.Cell(idx0, 1 , :) = Ini_LastBatch.Cell( batchsize , idx1 , :); 
end



% S1 = [1:Ntimesteps] + ([1:batchsize]'-1);
% S2 = [1:Ntimesteps] + ([1:batchsize]'-1) + batchsize;
% V2_priorep = 10.*S2-0.1;
% V2 = 10.*S2;
% V1 = 10.*S1;
% Ini_PriorEpochThisbatch = struct('Hidden',V2_priorep,'Cell',V2_priorep);
% Ini_LastBatch = struct('Hidden',V1,'Cell',V1);

end