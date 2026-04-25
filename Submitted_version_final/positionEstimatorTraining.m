function modelParameters = positionEstimatorTraining(training_data, opts)
    if nargin < 2, opts = struct(); end

    tic;

    num_dir     = size(training_data, 2);
    num_trials  = size(training_data, 1);
    num_neurons = size(training_data(1,1).spikes, 1);

    dt_class = get_opt(opts, 'dt_class', 80);
    dt_reg   = get_opt(opts, 'dt_reg', 20);
    M_pca_in = get_opt(opts, 'M_pca', []);
    M_lda_in = get_opt(opts, 'M_lda', []);
    use_lda  = get_opt(opts, 'use_lda', true);
    r_pcr    = get_opt(opts, 'r_pcr', []);
    global_decoder = get_opt(opts, 'global_decoder', false);

    minT = inf;
    for k = 1:num_dir
        for i = 1:num_trials
            % Find the absolute shortest trial length across all directions. 
            % We need this to make sure we don't accidentally index out of bounds later.
            minT = min(minT, size(training_data(i,k).spikes, 2));
            minT = min(minT, size(training_data(i,k).handPos, 2));
        end
    end

    class_times_default = [320, 400, 480, 560];
    class_times = get_opt(opts, 'class_times', class_times_default);
    class_times = class_times(class_times <= minT);

    reg_times_default = 320:dt_reg:min(560, minT);
    reg_times = get_opt(opts, 'reg_times', reg_times_default);

    modelParameters.class_times   = class_times;
    modelParameters.reg_times     = reg_times;
    modelParameters.num_dir       = num_dir;
    modelParameters.num_neurons   = num_neurons;
    modelParameters.dt_class      = dt_class;
    modelParameters.dt_reg        = dt_reg;
    modelParameters.global_decoder = global_decoder;
    modelParameters.use_lda       = use_lda;

    % Calculate the mean starting hand position (anchor point) for each direction.
    % This is super important because our PCR only predicts displacement from the start, 
    % so we need a good baseline to anchor the trajectory.
    mean_s = zeros(num_dir, 2);
    for k = 1:num_dir
        x_s = zeros(num_trials, 1);
        y_s = zeros(num_trials, 1);
        for i = 1:num_trials
            x_s(i) = training_data(i, k).handPos(1, 1);
            y_s(i) = training_data(i, k).handPos(2, 1);
        end
        mean_s(k, 1) = mean(x_s);
        mean_s(k, 2) = mean(y_s);
    end
    modelParameters.mean_starts = mean_s;
    modelParameters.global_mean_start = mean(mean_s, 1);

    % Classifiers (one per class_time)
    for i = 1:length(class_times)
        t = class_times(i);
        modelParameters.classifier(i) = train_pca_lda_classifier( ...
            training_data, t, dt_class, M_pca_in, M_lda_in, use_lda);
    end

    if global_decoder
        modelParameters.expert(1) = train_pcr_global(training_data, reg_times, dt_reg, r_pcr);
    else
        for k = 1:num_dir
            modelParameters.expert(k) = train_pcr(training_data(:, k), reg_times, dt_reg, r_pcr);
        end
    end

    modelParameters.current_dir      = [];
    modelParameters.current_class_id = 0;
    modelParameters.is_initialized   = false;

    modelParameters.train_time_sec = toc;
    fprintf('Train Time: %.2f s\n', modelParameters.train_time_sec);
end

function v = get_opt(opts, name, default)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end

function expert = train_pcr(dir_trials, reg_times, dt, r_pcr)

    num_trials = size(dir_trials, 1);
    num_neurons = size(dir_trials(1).spikes, 1);

    expert.dt = dt;
    expert.num_neurons = num_neurons;
    expert.reg_times = reg_times;

    for t_idx = 1:length(reg_times)
        t = reg_times(t_idx);
        num_bins = floor(t / dt);

        X = zeros(num_trials, num_neurons * num_bins);
        Y = zeros(num_trials, 2);

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

        % Use 'econ' SVD because our feature matrix X is usually way wider 
        % than the number of trials. It computes much faster and saves memory.
        [U, S, V] = svd(Xc, 'econ');

        % The max rank we can reliably use is bounded by the number of trials minus 1.
        rmax = min(size(S, 1), num_trials - 1);
        rmax = max(rmax, 1);
        if isempty(r_pcr)
            r = rmax;
        else
            r = min(r_pcr, rmax);
        end
        r = max(r, 1);

        Ur = U(:, 1:r);
        Sr = S(1:r, 1:r);
        Vr = V(:, 1:r);

        Beta = Vr * diag(1 ./ diag(Sr)) * Ur' * Yc;

        expert.time(t_idx).t        = t;
        expert.time(t_idx).num_bins = num_bins;
        expert.time(t_idx).mx       = mx;
        expert.time(t_idx).my       = my;
        expert.time(t_idx).Beta     = Beta;
        expert.time(t_idx).r_used   = r;
    end
end

function expert = train_pcr_global(train_data, reg_times, dt, r_pcr)

    num_dir = size(train_data, 2);
    num_trials = size(train_data, 1);
    num_neurons = size(train_data(1,1).spikes, 1);
    n_all = num_dir * num_trials;

    expert.dt = dt;
    expert.num_neurons = num_neurons;
    expert.reg_times = reg_times;

    for t_idx = 1:length(reg_times)
        t = reg_times(t_idx);
        num_bins = floor(t / dt);

        X = zeros(n_all, num_neurons * num_bins);
        Y = zeros(n_all, 2);

        row = 1;
        for k = 1:num_dir
            for i = 1:num_trials
                spikes = train_data(i, k).spikes(:, 1:t);
                feature = zeros(1, num_neurons * num_bins);
                for b = 1:num_bins
                    cols = (b-1)*dt + 1 : b*dt;
                    rows = (b-1)*num_neurons + 1 : b*num_neurons;
                    feature(rows) = sum(spikes(:, cols), 2)';
                end
                X(row, :) = feature;
                Y(row, :) = train_data(i, k).handPos(1:2, t)';
                row = row + 1;
            end
        end

        mx = mean(X, 1);
        my = mean(Y, 1);
        Xc = X - mx;
        Yc = Y - my;

        [U, S, V] = svd(Xc, 'econ');

        rmax = min(size(S, 1), n_all - 1);
        rmax = max(rmax, 1);
        if isempty(r_pcr)
            r = rmax;
        else
            r = min(r_pcr, rmax);
        end
        r = max(r, 1);

        Ur = U(:, 1:r);
        Sr = S(1:r, 1:r);
        Vr = V(:, 1:r);

        Beta = Vr * diag(1 ./ diag(Sr)) * Ur' * Yc;

        expert.time(t_idx).t        = t;
        expert.time(t_idx).num_bins = num_bins;
        expert.time(t_idx).mx       = mx;
        expert.time(t_idx).my       = my;
        expert.time(t_idx).Beta     = Beta;
        expert.time(t_idx).r_used   = r;
    end
end

function classifier = train_pca_lda_classifier(train_data, t, dt, M_pca_in, M_lda_in, use_lda)
%TRAIN_PCA_LDA_CLASSIFIER PCA + LDA direction classifier on binned spikes up to time t.

num_dir = size(train_data, 2);
num_trials = size(train_data, 1);
total_samples = num_dir * num_trials;
num_bins = floor(t / dt);
num_neurons = size(train_data(1,1).spikes, 1);
feature_dim = num_neurons * num_bins;

X = zeros(feature_dim, total_samples);
y = zeros(1, total_samples);
idx = 1;
for k = 1:num_dir
    for i = 1:num_trials
        spikes = train_data(i, k).spikes(:, 1:t);
        feature = zeros(feature_dim, 1);
        for b = 1:num_bins
            cols = (b-1)*dt + 1 : b*dt;
            rows = (b-1)*num_neurons + 1 : b*num_neurons;
            feature(rows) = sum(spikes(:, cols), 2);
        end
        X(:, idx) = feature;
        y(idx) = k;
        idx = idx + 1;
    end
end

mx = mean(X, 2);
Xc = X - mx;

% Again, using 'econ' SVD for PCA to quickly extract the top variance components.
[U, ~, ~] = svd(Xc, 'econ');
M_pca_max = size(U, 2);
if isempty(M_pca_in), M_pca = min(M_pca_max, 150);
else, M_pca = min(M_pca_max, M_pca_in); end
Wpca = U(:, 1:M_pca);
Xpca = Wpca' * Xc;

if use_lda
    mu_all = mean(Xpca, 2);
    Sw = zeros(M_pca, M_pca); Sb = zeros(M_pca, M_pca);
    for k = 1:num_dir
        Xk = Xpca(:, y == k); muk = mean(Xk, 2); Xkc = Xk - muk;
        Sw = Sw + Xkc * Xkc';
        dmu = muk - mu_all; Sb = Sb + size(Xk, 2) * (dmu * dmu');
    end
    % Add a tiny bit of ridge (identity matrix) to the within-class scatter matrix.
    % Classic trick to prevent the matrix from becoming singular/non-invertible.
    Sw = Sw + 1e-6 * eye(M_pca);
    [Wlda, D] = eig(Sw \ Sb);
    [~, order] = sort(real(diag(D)), 'descend');
    if isempty(M_lda_in), M_lda = min(num_dir - 1, size(Wlda, 2));
    else, M_lda = min([num_dir - 1, size(Wlda, 2), M_lda_in]); end
    M_lda = max(M_lda, 1);
    Wlda = real(Wlda(:, order(1:M_lda)));
    Wopt = Wpca * Wlda;
else
    Wopt = Wpca;
end

Xproj = Wopt' * Xc;
centroids = zeros(size(Xproj, 1), num_dir);
for k = 1:num_dir
    centroids(:, k) = mean(Xproj(:, y == k), 2);
end

classifier.t          = t;
classifier.dt         = dt;
classifier.num_bins   = num_bins;
classifier.num_neurons = num_neurons;
classifier.mx         = mx;
classifier.Wopt       = Wopt;
classifier.centroids  = centroids;
classifier.use_lda    = use_lda;
end
