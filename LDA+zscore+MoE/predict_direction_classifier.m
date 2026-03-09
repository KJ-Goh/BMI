function pred_dir = predict_direction_classifier(spikes_320, classifier)
    x = sum(spikes_320, 2);   % 98 x 1
    
     % z-score using training statistics
    x = (x - classifier.feature_mean) ./ classifier.feature_std;

    num_dir = size(classifier.mu, 2);
    g = zeros(1, num_dir);

    for k = 1:num_dir
        mu_k = classifier.mu(:, k);
        g(k) = mu_k' * classifier.invSigma * x - 0.5 * mu_k' * classifier.invSigma * mu_k + log(classifier.prior(k) + eps);
    end

    [~, pred_dir] = max(g);
end