 function [x, P] = kalman(x, P, z, m)
    % Predict
    x = m.A * x;
    P = m.A * P * m.A' + m.Q;
    % Update
    K = (P * m.C') / (m.C * P * m.C' + m.R);
    x = x + K * (z - m.C * x);
    P = P - K * m.C * P;
end