function [x, y, model] = position_estimator(test_data, model)
    T = size(test_data.spikes, 2);% How many time point we got
    
    if T == 320
        model.x = model.m0; %hidden state vector
        model.P = eye(5) * 0.1; %Covariance Matrix
        
        bins = 320/20;
        for i = 1:bins
            z = sum(test_data.spikes(:, (i-1)*20+1:i*20), 2);
            [model.x, model.P] = kalman(model.x, model.P, z, model);
        end
    else
        % 20ms update
        z = sum(test_data.spikes(:, end-19:end), 2); % The last 20 sampling points
        [model.x, model.P] = kalman(model.x, model.P, z, model);
    end
    
    x = model.x(1); 
    y = model.x(2);
end