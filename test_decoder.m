clear; 
clc; 
close all;

load('monkeydata_training.mat'); 
rng(42); 

colors = lines(8); 
[num_trials, num_dir] = size(trial);
n_train = round(0.8*num_trials); % 80% training set
conf_mat = zeros(num_dir, num_dir); % confusion matrix

for d = 1:num_dir % for each direction
    idx = randperm(num_trials);
    train_data(:,d) = trial(idx(1:n_train), d);
    test_data(:,d)  = trial(idx(n_train+1:end), d);
end

% Training model
fprintf('Training...\n');
tic;
modelParameters = positionEstimatorTraining(train_data);
fprintf('Train Time: %.2f s\n', toc);

fprintf('Testing...\n');
RMSE = 0; 
n_p = 0; % number of points
rmse_dir = zeros(1, size(test_data, 2));
np_dir = zeros(1, size(test_data, 2));
classifier_correct = 0;
classifier_total = 0;

figure('Color','w','Name','Multi-direction Trajectories');
hold on; 
grid on;

for angle = 1:size(test_data, 2)
    for tr = 1:size(test_data, 1)
        current = test_data(tr, angle);
        T_end = size(current.spikes, 2);

        % -------- direction classification for evaluation only --------
        spikes_320 = current.spikes(:, 1:320);
        pred_dir = predict_direction_classifier_local(spikes_320, modelParameters.classifier);
        conf_mat(angle, pred_dir) = conf_mat(angle, pred_dir) + 1;

        if pred_dir == angle
            classifier_correct = classifier_correct + 1;
        end
        classifier_total = classifier_total + 1;

        % -------- decoder test --------
        ex = []; 
        ey = []; 
        tx = []; 
        ty = [];

        model_now = modelParameters;

        % First decoded point at 320 ms
        test_data_320.spikes = current.spikes(:, 1:320);
        [x, y, model_now] = positionEstimator(test_data_320, model_now);

        ex = [ex, x];
        ey = [ey, y];
        tx = [tx, current.handPos(1,320)/10];
        ty = [ty, current.handPos(2,320)/10];

        % Continue decoding every 20 ms
        for t = 340:20:T_end
            test_data_t.spikes = current.spikes(:, 1:t);
            [x, y, model_now] = positionEstimator(test_data_t, model_now);

            ex = [ex, x];
            ey = [ey, y];
            tx = [tx, current.handPos(1,t)/10];
            ty = [ty, current.handPos(2,t)/10];
        end
        
        err_now = sum((ex-tx).^2 + (ey-ty).^2);
        RMSE = RMSE + err_now;
        n_p = n_p + length(ex);
        
        rmse_dir(angle) = rmse_dir(angle) + err_now;
        np_dir(angle) = np_dir(angle) + length(ex);
        
        if tr == 1 
            plot(tx, ty, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off'); 
            p = plot(ex, ey, 'Color', colors(angle,:), 'LineWidth', 1.5);
            p.DisplayName = ['Angle ' num2str(angle)];
        end
    end
end

xlabel('X Position (cm)'); 
ylabel('Y Position (cm)');
title('Hand Trajectory Decoding');
legend('Location', 'northeastoutside');
set(gca,'LineWidth',1.2,'FontSize',12);
axis equal; 
grid on;
exportgraphics(gcf,'Hand Trajectory Decoding.png','Resolution',600);

figure('Color','w');
imagesc(conf_mat);
colormap(parula); 
colorbar;
xlabel('Predicted Direction');
ylabel('True Direction');
title('Confusion Matrix');
xticks(1:num_dir);
yticks(1:num_dir);
axis equal tight;

for i = 1:num_dir
    for j = 1:num_dir
        text(j, i, num2str(conf_mat(i,j)), ...
            'HorizontalAlignment','center', ...
            'Color','white', ...
            'FontSize',15);
    end
end

fprintf('Overall RMSE: %.4f cm\n', sqrt(RMSE/n_p));
fprintf('Classification Accuracy: %.2f%%\n', 100 * classifier_correct / classifier_total);

for k = 1:length(rmse_dir)
    fprintf('Direction %d RMSE: %.4f cm\n', k, sqrt(rmse_dir(k)/np_dir(k)));
end

function pred_dir = predict_direction_classifier_local(spikes_320, classifier)
    x = sum(spikes_320, 2);   % 98 x 1

    % z-score using training statistics
    x = (x - classifier.feature_mean) ./ classifier.feature_std;
    x = classifier.W' * x;

    num_dir = size(classifier.mu, 2);
    g = zeros(1, num_dir);

    for k = 1:num_dir
        mu_k = classifier.mu(:, k);
        g(k) = mu_k' * classifier.invSigma * x ...
             - 0.5 * mu_k' * classifier.invSigma * mu_k ...
             + log(classifier.prior(k) + eps);
    end

    [~, pred_dir] = max(g);
end