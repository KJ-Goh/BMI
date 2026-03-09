function pred_dir = predict_direction_classifier(spikes_320, clf)
    x = sum(spikes_320, 2);   % 98 x 1

    num_dir = size(clf.mean_feat, 2);
    distance = zeros(1, num_dir);

    for k = 1:num_dir
        diff_k = x - clf.mean_feat(:, k);
        distance(k) = sum(diff_k.^2);
    end

    [~, pred_dir] = min(distance);
end