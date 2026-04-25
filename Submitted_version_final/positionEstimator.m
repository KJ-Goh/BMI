function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

    t = size(test_data.spikes, 2);
    class_times = modelParameters.class_times;
    reg_times = modelParameters.reg_times;

    dt_class = modelParameters.classifier(1).dt;
    if isfield(modelParameters, 'dt_class')
        dt_class = modelParameters.dt_class;
    end

    oracle = isfield(modelParameters, 'oracle_mode') && modelParameters.oracle_mode;
    global_dec = isfield(modelParameters, 'global_decoder') && modelParameters.global_decoder;

    if oracle
        modelParameters.current_dir = modelParameters.true_direction;
    else
        % We only update our direction guess at specific "classification checkpoints" (e.g., 320ms).
        % Otherwise, we just keep using whatever direction we predicted last time.
        idx_class = find(class_times == t, 1);
        if ~isempty(idx_class)
            classifier = modelParameters.classifier(idx_class);

            num_bins = floor(t / dt_class);
            num_neurons = size(test_data.spikes, 1);
            feature = zeros(num_neurons * num_bins, 1);
            for b = 1:num_bins
                cols = (b-1)*dt_class + 1 : b*dt_class;
                rows = (b-1)*num_neurons + 1 : b*num_neurons;
                feature(rows) = sum(test_data.spikes(:, cols), 2);
            end

            f_lda = classifier.Wopt' * (feature - classifier.mx);
            num_dir_c = size(classifier.centroids, 2);
            distance = zeros(1, num_dir_c);
            for k = 1:num_dir_c
                diff = f_lda - classifier.centroids(:, k);
                distance(k) = sum(diff .^ 2);
            end
            % Simple nearest-centroid rule in the PCA-LDA subspace. 
            % Very fast for real-time inference during the test phase.
            [~, pred_dir_clf] = min(distance);
            modelParameters.current_dir = pred_dir_clf;
        end

        if ~isfield(modelParameters, 'current_dir') || isempty(modelParameters.current_dir)
            modelParameters.current_dir = 1;
        end
    end

    pred_dir = modelParameters.current_dir;

    if global_dec
    expert_bank = modelParameters.expert(1);
else
    expert_bank = modelParameters.expert(pred_dir);
end

if isfield(modelParameters, 'dt_reg') && ~isempty(modelParameters.dt_reg)
    dt_reg = modelParameters.dt_reg;
else
    dt_reg = 20;
end

% Cap the time so we don't look for an expert beyond our max trained time.
% Then find the closest expert model trained for a time <= our current time t.
t_model = min(t, reg_times(end));
t_idx = find(reg_times <= t_model, 1, 'last');
if isempty(t_idx)
    t_idx = 1;
end

expert = expert_bank.time(t_idx);

if isfield(expert, 't') && ~isempty(expert.t)
    t_used = expert.t;
else
    t_used = reg_times(t_idx);
end

num_bins = floor(t_used / dt_reg);
num_neurons = size(test_data.spikes, 1);
feature = zeros(1, num_neurons * num_bins);

for b = 1:num_bins
    cols = (b-1)*dt_reg + 1 : b*dt_reg;
    rows = (b-1)*num_neurons + 1 : b*num_neurons;
    feature(rows) = sum(test_data.spikes(:, cols), 2)';
end

    y_pred = (feature - expert.mx) * expert.Beta + expert.my;

    if oracle
        dir_anchor = modelParameters.true_direction;
    else
        dir_anchor = pred_dir;
    end
    
    % The PCR model outputs a raw trajectory offset from the training set's mean start position.
    % To get the real (x, y) coordinates, we re-anchor it using the actual start hand position 
    % from this specific test trial. Crucial step, otherwise trajectories float randomly!
    mean_start = modelParameters.mean_starts(dir_anchor, :);
    shift = test_data.startHandPos' - mean_start;
    x = y_pred(1) + shift(1);
    y = y_pred(2) + shift(2);
end
