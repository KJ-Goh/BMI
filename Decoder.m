
clear; 
clc; 
close all;
load('monkeydata_training.mat'); 

colors = lines(8); 
[num_trials, num_dir] = size(trial);
n_train = round(0.8*num_trials);% 80% training set

for d = 1:num_dir %for each direction
    idx = randperm(num_trials);
    train_data(:,d) = trial(idx(1:n_train), d);
    test_data(:,d)  = trial(idx(n_train+1:end), d);
end

%Training model
fprintf('Training...\n');
tic;
model = position_estimator_training(train_data);
fprintf('Train Time: %.2f s\n', toc);

fprintf('Testing...\n');
RMSE = 0; 
n_p = 0; %number of point
figure('Color','w','Name','Multi-direction Trajectories');
hold on; 
grid on;

%tr:try  cur:current
for angle = 1:size(test_data, 2)
    for tr = 1:size(test_data, 1)
        cur = test_data(tr, angle);
        T_end = size(cur.spikes, 2);
        
        %e:expect t:true
        ex = []; ey = []; tx = []; ty = [];
        
        % Start from 320ms,20ms windows
        for t = 320:20:T_end
            chunk.spikes = cur.spikes(:, 1:t);
            [px, py, model] = position_estimator(chunk, model);
            
            ex = [ex, px]; 
            ey = [ey, py];
            tx = [tx, cur.handPos(1,t)/10]; 
            ty = [ty, cur.handPos(2,t)/10];
        end
        
        RMSE = RMSE + sum((ex-tx).^2 + (ey-ty).^2);
        n_p = n_p + length(ex);
        
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
axis equal; 
grid on;
exportgraphics(gcf,'Hand Trajectory Decoding.png','Resolution',600);
fprintf('RMSE: %.4f cm\n', sqrt(RMSE/n_p));
