function classifier = train_direction_classifier(train_data)
    num_dir = size(train_data,2); %number of directions
    num_trials = size(train_data, 1);%number of trials
    num_neurons = size(train_data(1,1).spikes,1);

    % Collect all features
    total_samples = num_dir * num_trials;

    % Preallocate
    X = zeros(num_neurons , total_samples);
    y = zeros(1 , total_samples);
    idx = 1;

    for k = 1:num_dir
        for i = 1:num_trials
            s = train_data(i,k).spikes(:,1:320); %kth direction, ith trial
            feature = sum(s, 2);   % 98 x 1
            X(:,idx) = feature;
            y(idx) = k; %direction k

            idx = idx + 1;
        end
    end
    
    %z-score
    classifier.feature_mean = mean(X, 2);
    classifier.feature_std = std(X, 0, 2) + 1e-8;

    Xn = (X - classifier.feature_mean) ./ classifier.feature_std;
    % Compute class means
    classifier.mu = zeros(num_neurons, num_dir);
    classifier.prior = zeros(1, num_dir);
    
    %Mean for each direction
    for k = 1:num_dir
        Xk = Xn(:, y==k);
        classifier.mu(:,k) = mean(Xk, 2);
        classifier.prior(k) = size(Xk, 2) / size(Xn, 2);
    end
    
    %Pooled covariance
    Sigma = zeros(num_neurons, num_neurons);
    for k = 1:num_dir
        Xk = Xn(:,y==k);
        Xc = Xk - classifier.mu(:,k);
        Sigma = Sigma + Xc * Xc';
    end

    Sigma = Sigma / size(Xn,2);

    % regularization
    reg = 1e-3;
    Sigma = Sigma + reg * eye(num_neurons);

    classifier.Sigma = Sigma;
    classifier.invSigma = inv(Sigma);

end



