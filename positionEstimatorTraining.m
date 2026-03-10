function modelParameters = positionEstimatorTraining(training_data)
    fprintf('Training...\n');
    tic;
    modelParameters = train_moe_model(training_data);
    fprintf('Train Time: %.2f s\n', toc);
end

function expert = train_kalman_expert(dir_trials)
    dt = 20; 
    % State Matrix X [x; y; vx; vy; 1] & Observation matrix Z
    X = []; 
    Z = [];
    X0 = [];   % save initial state of each trial
    
    for t = 1:size(dir_trials,1) %Each direction multiple trial
            p = dir_trials(t).handPos(1:2, :)/10; %2*T real position, mm to cm
            s = dir_trials(t).spikes; %98*T Neurons
            
            num_bins = floor(size(s, 2) / dt); % how many 20ms bins we have
            z_bin = zeros(98, num_bins); % #spikes in each neuron
            p_bin = zeros(2, num_bins); % Hand Position
            
            for b = 1:num_bins
                idx = (b-1)*dt + 1 : b*dt; % 1:20,21:40,41:60...
                %The total number of spikes of each neuron in the b-th 20ms period
                z_bin(:, b) = sum(s(:, idx), 2); 
                %The hand position (2D) at the b-th 20ms moment
                p_bin(:, b) = p(:, idx(end)); 
            end
            
            %A speed sequence of 2×num_bins
            v_bin = [[0;0], diff(p_bin, 1, 2) / (dt/1000)];
            x_seq = [p_bin; v_bin; ones(1, num_bins)];
            
            X = [X, [p_bin; v_bin; ones(1, num_bins)]];
            Z = [Z, z_bin];
            X0 = [X0, x_seq(:,1)];
    end


    X1 = X(:, 1:end-1); %Current state
    X2 = X(:, 2:end); %Next State
    expert.A = X2 / X1; %X2≈AX1, least error
    expert.C = Z / X;%Z≈CX, Given position and velocity, how many spikes for each neuron. What is the relationship between the firing rate of neuron i and the position/velocity of the hand?
    expert.Q = (X2 - expert.A*X1)*(X2 - expert.A*X1)' / size(X1,2) + eye(5)*1e-4; %How inaccuracte the model. Covariance of error
    expert.R = (Z - expert.C*X)*(Z - expert.C*X)' / size(X,2) + eye(98)*1e-2; %How nosiy the spike. 
    expert.m0 = mean(X0, 2);
    expert.P0 = eye(5) * 0.1;
end

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
end