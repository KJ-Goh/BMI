function pred_dir = predict_direction_classifier(spikes_320, classifier)
    r = causal_estimator(spikes_320, classifier.alpha);
    x = r(:, end);

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