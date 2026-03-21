function modelParameters = positionEstimatorTraining(training_data)
    tic;
    num_dir = size(training_data, 2);
    num_trials = size(training_data, 1);
    class_times = [320, 400, 480, 560];
    reg_times = 320:20:560;

    % This stores the average starting position for each direction
    % Later use it to do a simple shift correction
    mean_s = zeros(num_dir, 2);
    for k = 1:num_dir
        x_s = zeros(num_trials, 1);
        y_s = zeros(num_trials, 1);
        for i = 1:num_trials
            x_s(i) = training_data(i, k).handPos(1, 1);
            y_s(i) = training_data(i, k).handPos(2, 1);
        end
        mean_s(k, 1) = mean(x_s); %Average start x
        mean_s(k, 2) = mean(y_s); %Average start y
    end
    %store the parameter
    modelParameters.mean_starts = mean_s;
    modelParameters.class_times = class_times;
    modelParameters.reg_times = reg_times;

    % Train one classifier at each decision time
    % We got f320,f400,f480,f560
    for i = 1:length(class_times)
        t = class_times(i);
        modelParameters.classifier(i) = train_pca_lda(training_data, t); 
    end

    % Train one PCR expert for each direction
    % We got k experts for k direction
    for k = 1:num_dir
        modelParameters.expert(k) = train_pcr(training_data(:, k), reg_times);
    end
    fprintf('Train Time: %.2f s\n', toc);
end

function classifier = train_pca_lda(train_data, t)
    num_dir = size(train_data, 2);
    num_trials = size(train_data, 1);
    total_samples = num_dir * num_trials;
    dt = 80; %time window
    num_bins = floor(t / dt); %how many bins
    num_neurons = size(train_data(1,1).spikes, 1);
    feature_dim = num_neurons * num_bins;
    X = zeros(feature_dim, total_samples);%feaure matrix
    y = zeros(1, total_samples);%label vector

    % Build one long feature vector by stacking 80 ms spike counts
    idx = 1;
    %for each direction and trials,take the spike data and build an empty
    %feature vector
    for k = 1:num_dir 
        for i = 1:num_trials 
            spikes = train_data(i, k).spikes(:, 1:t);
            feature = zeros(feature_dim, 1);
            for b = 1:num_bins %for each 80ms bin, how many spikes happened
                cols = (b-1)*dt + 1 : b*dt;
                rows = (b-1)*num_neurons + 1 : b*num_neurons;
                feature(rows) = sum(spikes(:, cols), 2); % stack all bin together
            end
            X(:, idx) = feature;
            y(idx) = k; %direction
            idx = idx + 1;
        end
    end
    % Center the data first
    mx = mean(X, 2);
    Xc = X - mx;

    % PCA using SVD
    [U, ~, ~] = svd(Xc, 'econ');%col vector of U is the Main component direction
    % Keep enough PCs, but not too many
    M_pca = min(size(U, 2) - num_dir, 150);
    if M_pca < 1
        M_pca = size(U, 2);
    end
    Wpca = U(:, 1:M_pca);
    Xpca = Wpca' * Xc;%projection

    % LDA part
    mu_all = mean(Xpca, 2);
    Sw = zeros(M_pca, M_pca);%within-class scatter
    Sb = zeros(M_pca, M_pca);%between-class scatter
    for k = 1:num_dir
        Xk = Xpca(:, y == k);
        muk = mean(Xk, 2);
        Xkc = Xk - muk;%X_i - mu_k
        Sw = Sw + Xkc * Xkc';
        dmu = muk - mu_all; %mu_k-mu
        Sb = Sb + size(Xk, 2) * (dmu * dmu');
    end
    % Small regularization just to make things more stable
    Sw = Sw + 1e-6 * eye(M_pca);

    [Wlda, D] = eig(Sw \ Sb);
    [~, order] = sort(real(diag(D)), 'descend');

    % For 8 directions, LDA gives at most 7 useful dimensions
    M_lda = min(num_dir - 1, size(Wlda, 2));
    Wlda = real(Wlda(:, order(1:M_lda)));
    Wopt = Wpca * Wlda;
    % Save class centroids in the final projected space
    Xlda = Wopt' * Xc;
    centroids = zeros(M_lda, num_dir);
    for k = 1:num_dir
        centroids(:, k) = mean(Xlda(:, y == k), 2);
    end
    classifier.Wopt = Wopt;
    classifier.mx = mx;
    classifier.centroids = centroids;
end

function expert = train_pcr(dir_trials, reg_times)
    dt = 20;%time windows
    num_trials = size(dir_trials, 1);
    num_neurons = size(dir_trials(1).spikes, 1);
    %for each reg_times
    for t_idx = 1:length(reg_times)
        t = reg_times(t_idx);
        num_bins = floor(t / dt);
        X = zeros(num_trials, num_neurons * num_bins);%input features
        Y = zeros(num_trials, 2);%target position
        for i = 1:num_trials
            spikes = dir_trials(i).spikes(:, 1:t);
            feature = zeros(1, num_neurons * num_bins);
            for b = 1:num_bins
                cols = (b-1)*dt + 1 : b*dt;
                rows = (b-1)*num_neurons + 1 : b*num_neurons;
                feature(rows) = sum(spikes(:, cols), 2)';
            end
            X(i, :) = feature;
            Y(i, :) = dir_trials(i).handPos(1:2, t)';
        end

        mx = mean(X, 1);
        my = mean(Y, 1);
        Xc = X - mx;
        Yc = Y - my;

        % PCR is just SVD + linear regression in PC space
        [U, S, V] = svd(Xc, 'econ');%V:principal directions of the feature space. S: how important is the direction
        r = min(num_trials - 1, size(S, 1));%how many should be keep
        Ur = U(:, 1:r);
        Sr = S(1:r, 1:r);
        Vr = V(:, 1:r);
        Beta = Vr * diag(1 ./ diag(Sr)) * Ur' * Yc;
        expert.time(t_idx).Beta = Beta;
        expert.time(t_idx).mx = mx;
        expert.time(t_idx).my = my;
    end
end