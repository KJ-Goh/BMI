function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
    t = size(test_data.spikes, 2);
    class_times = modelParameters.class_times;
    reg_times = modelParameters.reg_times;

    % Update direction only at the chosen classification times
    idx_class = find(class_times == t, 1);
    if ~isempty(idx_class) %if the time is class time
        classifier = modelParameters.classifier(idx_class);
    %same with training
        dt_class = 80;
        num_bins = floor(t / dt_class);
        num_neurons = size(test_data.spikes, 1);
        feature = zeros(num_neurons * num_bins, 1);
        for b = 1:num_bins
            cols = (b-1)*dt_class + 1 : b*dt_class;
            rows = (b-1)*num_neurons + 1 : b*num_neurons;
            feature(rows) = sum(test_data.spikes(:, cols), 2);
        end

        % Project into the PCA-LDA space
        f_lda = classifier.Wopt' * (feature - classifier.mx);%z=W_opt(x-mu)
        num_dir = size(classifier.centroids, 2);
        distance = zeros(1, num_dir);
        for k = 1:num_dir
            diff = f_lda - classifier.centroids(:, k); 
            distance(k) = sum(diff .^ 2);%Squared Euclidean distance
        end
        [~, pred_dir] = min(distance);%nearst distance
        modelParameters.current_dir = pred_dir;
    end

    % If direction has not been set yet, use a default one
    if ~isfield(modelParameters, 'current_dir')
        modelParameters.current_dir = 1;
    end
    pred_dir = modelParameters.current_dir;

    % Use the closest available regression model
    % If t goes beyond 560, just keep using the 560 ms model
    t_model = min(t, reg_times(end));
    t_idx = find(reg_times <= t_model, 1, 'last');
    expert = modelParameters.expert(pred_dir).time(t_idx);
    dt_reg = 20;
    t_use = reg_times(t_idx);
    num_bins = floor(t_use / dt_reg);
    num_neurons = size(test_data.spikes, 1);
    feature = zeros(1, num_neurons * num_bins);
    for b = 1:num_bins
        cols = (b-1)*dt_reg + 1 : b*dt_reg;
        rows = (b-1)*num_neurons + 1 : b*num_neurons;
        feature(rows) = sum(test_data.spikes(:, cols), 2)';
    end
    
    %y_predict = (x-mu_x)*B + mu_y
    y_pred = (feature - expert.mx) * expert.Beta + expert.my;

    % Simple anchor correction using the starting hand position
    mean_start = modelParameters.mean_starts(pred_dir, :);
    shift = test_data.startHandPos' - mean_start;
    x = y_pred(1) + shift(1);
    y = y_pred(2) + shift(2);
end