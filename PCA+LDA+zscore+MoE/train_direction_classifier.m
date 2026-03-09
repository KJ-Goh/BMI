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

    %PCA
    C = cov(Xn');
    [V,D] = eig(C);
    % sort eigenvalues
    [~, sort_idx] = sort(diag(D),'descend');
    V = V(:,sort_idx);
    % choose number of PCs
    pca_dim = 15;
    classifier.W = V(:,1:pca_dim);
    % project data
    Xp = classifier.W' * Xn;

    % Compute class means
    classifier.mu = zeros(pca_dim, num_dir);
    classifier.prior = zeros(1, num_dir);
    
    %Mean for each direction
    for k = 1:num_dir
        Xk = Xp(:, y==k);
        classifier.mu(:,k) = mean(Xk, 2);
        classifier.prior(k) = size(Xk, 2) / size(Xp, 2);
    end
    
    %Pooled covariance
    Sigma = zeros(pca_dim, pca_dim);
    for k = 1:num_dir
        Xk = Xp(:,y==k);
        Xc = Xk - classifier.mu(:,k);
        Sigma = Sigma + Xc * Xc';
    end

    Sigma = Sigma / size(Xp,2);

    % regularization
    reg = 1e-3;
    Sigma = Sigma + reg * eye(pca_dim);

    classifier.Sigma = Sigma;
    classifier.invSigma = Sigma \ eye(size(Sigma));

end



