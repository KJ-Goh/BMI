function model = train_moe_model(train_data)
    num_dir = size(train_data, 2);
    
    % Train direction classifier
    fprintf('Training direction classifier...\n');
    model.classifier = train_direction_classifier(train_data);

    % Train one Kalman expert per direction
    for k = 1:num_dir
        fprintf('Training expert for direction %d...\n', k);
        model.expert(k) = train_kalman_expert(train_data(:, k));
    end
    for k = 1:num_dir
        fprintf('Training expert for direction %d...\n', k);
        model.expert(k) = train_kalman_expert(train_data(:, k));
    end
end