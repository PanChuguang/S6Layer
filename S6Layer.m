classdef S6Layer < nnet.layer.Layer ...
         & nnet.layer.Formattable  % ...
        % & nnet.layer.Acceleratable % (Optional)

        % S6 layer for Mamba block construction
        % Refer to 10.1016/j.isatra.2026.09.020 (DOI)

    properties
        % Declare layer properties here.
        StateDim % State dimension size (d_state)
        Expand % Expand factor (expand)
        Conv1dKernelSize % 1D Convolution kernel size (d_conv)
    end

    properties (Learnable)
        % Declare learnable parameters here.
        InputProjWeights % [2*d_inner,d_model] where d_inner==d_model*expand, d_model==inputChannels "C"
        OutputProjWeights % [d_model,d_inner]

        Conv1dWeights % [d_conv,1,1,d_inner] filterSize-by-numChannelsPerGroup-by-numFiltersPerGroup-by-numGroups "TCUU"
        Conv1dBias % [1*d_inner,1] numFiltersPerGroup*numGroups-by-1 

        LogA % [d_inner,d_state] learnable state space parameter

        RawD % [d_inner,1] learnable state space parameter 

        TimeProjWeights % [d_inner,d_inner] 
        TimeProjBias % [d_inner,1]

        BProjWeights % [d_state*d_inner,d_inner] 

        CProjWeights % [d_state*d_inner,d_inner]

    end

    % properties (State)
    %     % Declare state parameters here.
    %     HiddenState % [d_inner,d_state] 
    % end

    % properties (Learnable, State)
    %     % (Optional) Nested dlnetwork objects with both learnable
    %     % parameters and state parameters.
    % 
    %     % Declare nested networks with learnable and state parameters here.
    % end

    methods
        function layer = S6Layer(stateDim,expand,kernelSize,nvargs)
            % This function must have the same name as the class.
            % Define layer constructor function here.

            arguments (Input)
                stateDim (1,1) {mustBePositive,mustBeInteger}
                expand (1,1) {mustBePositive,mustBeInteger}=2
                kernelSize (1,1) {mustBePositive,mustBeInteger}=4
                nvargs.Name {mustBeTextScalar}="S6Layer" 
            end

            layer.StateDim = stateDim;
            layer.Expand = expand;
            layer.Conv1dKernelSize = kernelSize;

            layer.Description = "S6 layer for building Mamba block";

            layer.Name = nvargs.Name;
        end

        function layer = initialize(layer,layout)
            %
            % Inputs:
            %         layer  - Layer to initialize
            %         layout - Data layout, specified as a networkDataLayout
            %                  object
            %
            % Outputs:
            %         layer - Initialized layer
            %
            %  - For layers with multiple inputs, replace layout with 
            %    layout1,...,layoutN, where N is the number of inputs.
            
            % Define layer initialization function here.

            % Find the dim of input (d_model)
            d_model = layout.Size(finddim(layout,"C"));

            d_inner = d_model * layer.Expand;
            d_state = layer.StateDim;

            % Initialize learnable parameters
            if isempty(layer.InputProjWeights)
                layer.InputProjWeights = dlarray(randn([2*d_inner d_model]));
            end

            if isempty(layer.OutputProjWeights)
                layer.OutputProjWeights = dlarray(randn([d_model d_inner]));
            end

            if isempty(layer.Conv1dWeights)
                layer.Conv1dWeights = dlarray(randn([layer.Conv1dKernelSize 1 1 d_inner]));
            end

            if isempty(layer.Conv1dBias)
                layer.Conv1dBias = dlarray(zeros([1*d_inner 1]));
            end

            if isempty(layer.LogA)
                logA = log(repmat(1:d_state,d_inner,1)) / d_state; % [d_inner,d_state]
                layer.LogA = dlarray(logA);
            end

            if isempty(layer.RawD)
                layer.RawD = dlarray(ones([d_inner 1])*0.1); % [d_inner,1]
            end

            if isempty(layer.TimeProjWeights)
                layer.TimeProjWeights = dlarray(randn([d_inner d_inner]));
            end

            if isempty(layer.TimeProjBias)
                layer.TimeProjBias = dlarray(zeros([d_inner 1]));
            end

            if isempty(layer.BProjWeights)
                layer.BProjWeights = dlarray(randn([d_state*d_inner d_inner]));
            end

            if isempty(layer.CProjWeights)
                layer.CProjWeights = dlarray(randn([d_state*d_inner d_inner]));
            end

            % % Initialize states. (No state)
            % if isempty(layer.HiddenState)
            %     layer.HiddenState = zeros([d_inner d_state]);
            % end
           
        end
        

        function Y = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Y     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Y with 
            %    Y1,...,YM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.

            batchSize = size(X,finddim(X,"B")); % batch size
            seqLen = size(X,finddim(X,"T")); % sequence length
            d_model = size(X,finddim(X,"C")); % input dim
            d_inner = size(layer.OutputProjWeights,2); % d_inner
            d_state = layer.StateDim; % d_state

            XZ = fullyconnect(X,layer.InputProjWeights,zeros([2*d_inner 1])); % [2*d_inner,B,T] "CBT"
            X = XZ(1:d_inner,:,:); % [d_inner,B,T] 
            Z = XZ(d_inner+1:end,:,:); % [d_inner,B,T]

            % Depthwise 1D-Conv
            X = dlconv(X,layer.Conv1dWeights,layer.Conv1dBias,"WeightsFormat","TCUU","Padding","causal"); % [d_inner,B,T] "CBT"

            % State space parameters
            dt = softplus(fullyconnect(X,layer.TimeProjWeights,layer.TimeProjBias)) + 1e-4; % [d_inner,B,T] "CBT"
            dt = clip(dt,-5,5); % [d_inner,B,T] "CBT"
            A = -exp(clip(layer.LogA,-5,5)); % [d_inner,d_state]
            D = clip(layer.RawD,0.1,2); % [d_inner,1]

            B = fullyconnect(X,layer.BProjWeights,zeros([d_state*d_inner 1])); % [d_state*d_inner,B,T] "CBT"
            C = fullyconnect(X,layer.CProjWeights,zeros([d_state*d_inner 1])); % [d_state*d_inner,B,T]

            B = reshape(B,d_inner,d_state,batchSize,seqLen); % [d_inner,d_state,B,T]
            C = reshape(C,d_inner,d_state,batchSize,seqLen); % [d_inner,d_state,B,T]

            %% Selective Scan
            dtA = exp(reshape(dt,d_inner,1,batchSize,seqLen) .* A); % [d_inner,d_state,B,T]
            dtB = reshape(dt,d_inner,1,batchSize,seqLen) .* B; % [d_inner,d_state,B,T]

            stateX = dlarray(zeros([d_inner d_state batchSize]));
            seqY = zeros([d_inner batchSize seqLen],"like",X);

            % Loop over time step
            for t = 1:seqLen
                stateX = dtA(:,:,:,t).*stateX + dtB(:,:,:,t);
                stateX = clip(stateX,-1e6,1e6);
                seqY(:,:,t) = sum(stateX.*C(:,:,:,t),2);
            end

            seqY = seqY + X.*D; % [d_inner,B,T] "CBT"

            Y = fullyconnect(seqY.*Z,layer.OutputProjWeights,zeros([d_model 1]));
        end

        % function [Y,state,memory] = forward(layer,X)
        %     % (Optional) Forward input data through the layer at training
        %     % time and output the result, the updated state, and a memory
        %     % value.
        %     %
        %     % Inputs:
        %     %         layer - Layer to forward propagate through 
        %     %         X     - Layer input data
        %     % Outputs:
        %     %         Y      - Output of layer forward function 
        %     %         state  - (Optional) Updated layer state 
        %     %         memory - (Optional) Memory value for custom backward
        %     %                  function
        %     %
        %     %  - For layers with multiple inputs, replace X with X1,...,XN, 
        %     %    where N is the number of inputs.
        %     %  - For layers with multiple outputs, replace Y with 
        %     %    Y1,...,YM, where M is the number of outputs.
        %     %  - For layers with multiple state parameters, replace state 
        %     %    with state1,...,stateK, where K is the number of state 
        %     %    parameters.
        % 
        %     % Define layer forward function here.
        % end

        % function layer = resetState(layer)
        %     % Define reset state function here.
        % end

        % function [dLdX,dLdW,dLdSin] = backward(layer,X,Y,dLdY,dLdSout,memory)
        %     % (Optional) Backward propagate the derivative of the loss
        %     % function through the layer.
        %     %
        %     % Inputs:
        %     %         layer   - Layer to backward propagate through 
        %     %         X       - Layer input data 
        %     %         Y       - Layer output data 
        %     %         dLdY    - Derivative of loss with respect to layer 
        %     %                   output
        %     %         dLdSout - (Optional) Derivative of loss with respect 
        %     %                   to state output
        %     %         memory  - Memory value from forward function
        %     % Outputs:
        %     %         dLdX   - Derivative of loss with respect to layer input
        %     %         dLdW   - (Optional) Derivative of loss with respect to
        %     %                  learnable parameter 
        %     %         dLdSin - (Optional) Derivative of loss with respect to 
        %     %                  state input
        %     %
        %     %  - For layers with state parameters, the backward syntax must
        %     %    include both dLdSout and dLdSin, or neither.
        %     %  - For layers with multiple inputs, replace X and dLdX with
        %     %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
        %     %    the number of inputs.
        %     %  - For layers with multiple outputs, replace Y and dLdY with
        %     %    Y1,...,YM and dLdY,...,dLdYM, respectively, where M is the
        %     %    number of outputs.
        %     %  - For layers with multiple learnable parameters, replace 
        %     %    dLdW with dLdW1,...,dLdWP, where P is the number of 
        %     %    learnable parameters.
        %     %  - For layers with multiple state parameters, replace dLdSin
        %     %    and dLdSout with dLdSin1,...,dLdSinK and 
        %     %    dLdSout1,...,dldSoutK, respectively, where K is the number
        %     %    of state parameters.
        % 
        %     % Define layer backward function here.
        % end
    end
end

%% Support Functions of S6 layer

function X = softplus(X)
% Numerically stable implementation of softplus: Z = log(1 + exp(X))
% When X < 0, Z = log(1 + exp(X))
% When X >=0, Z = X + log(1 + exp(-X))

X = max(X,0) + log(1 + exp(-abs(X))); % Refer to softplusLayer
end