function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

    t = size(test_data.spikes, 2);

    if t == 320
        % Predict direction from first 320 ms
        spikes_320 = test_data.spikes(:, 1:320);
        pred_dir = predict_direction_classifier(spikes_320, modelParameters.classifier);

        % Select expert using predicted direction
        expert = modelParameters.expert(pred_dir);

        x_state = expert.m0;
        P = expert.P0;

        for i = 1:(320/20)
            idx_start = (i-1)*20 + 1;
            idx_end = i*20;
            z = sum(test_data.spikes(:, idx_start:idx_end), 2);
            [x_state, P] = kalman(x_state, P, z, expert);
        end

        modelParameters.current_dir = pred_dir;
        modelParameters.current_expert = expert;
        modelParameters.current_state = x_state;
        modelParameters.current_P = P;

    elseif t > 320
        expert = modelParameters.current_expert;
        x_state = modelParameters.current_state;
        P = modelParameters.current_P;

        z = sum(test_data.spikes(:, t-19:t), 2);
        [x_state, P] = kalman(x_state, P, z, expert);

        modelParameters.current_state = x_state;
        modelParameters.current_P = P;
    end

    x = modelParameters.current_state(1);
    y = modelParameters.current_state(2);
end

function pred_dir = predict_direction_classifier(spikes_320, classifier)
    x = sum(spikes_320, 2);   % 98 x 1

     % z-score using training statistics
    x = (x - classifier.feature_mean) ./ classifier.feature_std;
    x = classifier.W' * x;
    num_dir = size(classifier.mu, 2);
    g = zeros(1, num_dir);

    for k = 1:num_dir
        mu_k = classifier.mu(:, k);
        g(k) = mu_k' * classifier.invSigma * x - 0.5 * mu_k' * classifier.invSigma * mu_k + log(classifier.prior(k) + eps);
    end

    [~, pred_dir] = max(g);
end

function [x, P] = kalman(x, P, z, m)
    % Predict
    x = m.A * x;
    P = m.A * P * m.A' + m.Q;
    % Update
    K = (P * m.C') / (m.C * P * m.C' + m.R);
    x = x + K * (z - m.C * x);
    P = P - K * m.C * P;
end