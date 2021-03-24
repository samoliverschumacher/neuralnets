Main algorithm **TrainSequnceNNetV3.m** accepts a combination of layers which are RNNs or normal fully connected layers, and infers how they are connected by looking at the settings defined when the individual layers are created.
Supported features;
- gradient clipping
- learning rules: SGD, SGD w Learning rate schedule, Momentum, Adam, ~~Adagrad~~, Adadelta
- Cost functions: Multiclass & Binary Cross-entropy, L2, L1, Weighted binary cross entropy
- Automatic weight initialiser selector (based on activation function)
- LSTM cell with peephole connections
- input sequence masking
- teacher forcing
- Attention mechanism for attentional encoder-decoders
- BiDirectional LSTM

To be added;
- Stacked LSTM support
- batch normalisation
- parralel computing support
- Other types of NNet layers: GRU, Convoutional, bayesian neural network!

The TrainSequenceNNetV3.m allows any combinnation of fully connected & LSTM layers to be input, it can (should) figure out how to connect them together. i.e. FC -> LSTM -> LSTM -> FC or LSTM -> FC, or LSTM -> LSTM. 
Will try to add examples of how it can be flexible to different configurations, and how to do so.

See the repository machinelearningprojects/toyProjects/lstmRomannumerals for a working example. 

This is purely a project for teaching and learning, it's been my education in LSTMs. Please feel free to use & improve!
FYI - if you just want to use a LSTM in matlab, there are toolboxes that do a much much better job!
