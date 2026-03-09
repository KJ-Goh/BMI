function r = causal_estimator(spikes, alpha)
    % spikes: N x T
    % r: N x T causal smoothed firing-rate proxy

    [N, T] = size(spikes);
    r = zeros(N, T);

    r(:,1) = alpha * spikes(:,1);

    for t = 2:T
        r(:,t) = (1 - alpha) * r(:,t-1) + alpha * spikes(:,t);
    end
end