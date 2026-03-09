function classifier = train_direction_classifier(train_data)
    num_dir = size(train_data, 2);
    num_neurons = size(train_data(1,1).spikes, 1);

    classifier.mean_feat = zeros(num_neurons, num_dir);

    for k = 1:num_dir
        feature_k = [];

        for i = 1:size(train_data, 1)
            s = train_data(i, k).spikes(:, 1:320);
            feat = sum(s, 2);   % 98 x 1
            feature_k = [feature_k, feat];
        end

        classifier.mean_feat(:, k) = mean(feature_k, 2);
    end
end