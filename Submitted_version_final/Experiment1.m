%RUN_PAPER_EXPERIMENTS
%   mode = 'quick'  ->  5 MC iters (Just for test)
%   mode = 'paper'  ->  50 MC iters 
%
%   Outputs -> experiment_outputs/
%     TableI_comparison.csv
%     Fig1_rmse_vs_r.png
%     Fig2_error_over_time.png
%     Fig3_trajectories_overview.png
%     main_results_bundle.mat

mode    = 'paper';
out_dir = './experiment_outputs/';
if ~exist(out_dir,'dir'), mkdir(out_dir); end
set(0,'DefaultAxesFontSize',11); set(0,'DefaultLineLineWidth',1.5);

cfg.sP_default   = 0.8;
cfg.times_decode = 320:20:560;  
cfg.rng_base     = 2013;
if strcmp(mode,'quick')
    cfg.num_mc_iters = 5;
    r_grid = [5, 20, 80, Inf];
else
    cfg.num_mc_iters = 50;
    r_grid = [5,10,15,20,30,40,50,60,80,Inf];
end
load('monkeydata_training.mat');   % -> trial
cols8 = lines(8);                  % consistent 8-direction colour scheme

%% Full-rank MC run (Table I + Fig 2 data)
fprintf('Full-rank MC (%d iters)...\n', cfg.num_mc_iters);
stats = MC_benchmarks(trial, cfg);

%% Table I
rel_imp = (stats.rmse_global_mean - stats.rmse_proposed_mean) / stats.rmse_global_mean * 100;
rel_imp_kf = (stats.rmse_kalman_mean - stats.rmse_proposed_mean) / stats.rmse_kalman_mean * 100;
fprintf('\n%-22s %10s %10s %14s\n','Model','Mean','Std','RelImp(%)');
fprintf('%-22s %10.4f %10.4f %14s\n',  'Mean Trajectory',          stats.rmse_avg_mean,     stats.rmse_avg_std,     '-');
fprintf('%-22s %10.4f %10.4f %14s\n',  'Global PCR',               stats.rmse_global_mean,  stats.rmse_global_std,  '-');
fprintf('%-22s %10.4f %10.4f %14s\n',  'Global Kalman Filter',     stats.rmse_kalman_mean,  stats.rmse_kalman_std,  '-');
fprintf('%-22s %10.4f %10.4f %14.2f\n','Proposed (PCA-LDA+Dir-PCR)',stats.rmse_proposed_mean,stats.rmse_proposed_std,rel_imp);
fprintf('%-22s %10.4f %10.4f %14s\n',  'Oracle Lower Bound',       stats.rmse_oracle_mean,  stats.rmse_oracle_std,  '-');
fprintf('Proposed vs Global Kalman Filter: %.2f%% relative improvement, Wilcoxon signed-rank p = %.3g (N=%d)\n', ...
    rel_imp_kf, stats.wilcoxon_proposed_vs_kalman_p, numel(stats.rmse_kalman));
T1 = table({'Mean_Trajectory';'Global_PCR';'Global_Kalman_Filter';'Proposed_PCA_LDA_DirPCR';'Oracle_Lower_Bound'}, ...
    [stats.rmse_avg_mean;stats.rmse_global_mean;stats.rmse_kalman_mean;stats.rmse_proposed_mean;stats.rmse_oracle_mean], ...
    [stats.rmse_avg_std; stats.rmse_global_std; stats.rmse_kalman_std; stats.rmse_proposed_std; stats.rmse_oracle_std], ...
    [0;0;0;rel_imp;0], ...
    [0;0;0;rel_imp_kf;0], ...
    'VariableNames',{'Model','Mean_RMSE_cm','Std_RMSE_cm','Relative_Improvement_vs_GlobalPCR_pct','Relative_Improvement_vs_GlobalKF_pct'});
writetable(T1, fullfile(out_dir,'TableI_comparison.csv'));
fprintf('Table I saved.\n');

%% Fig 1: RMSE vs r
n_r = length(r_grid);
r_prp_mean=zeros(1,n_r); r_prp_std=zeros(1,n_r);
r_orc_mean=zeros(1,n_r); r_orc_std=zeros(1,n_r);
r_glb_mean=zeros(1,n_r); r_glb_std=zeros(1,n_r);
for ri = 1:n_r
    rv = r_grid(ri);
    if isinf(rv), rp=[]; else, rp=rv; end
    fprintf('  r sweep %d/%d (r=%s)...\n',ri,n_r,num2str(rv));
    sr = MC_benchmarks(trial, cfg, struct('r_pcr',rp));
    r_prp_mean(ri)=sr.rmse_proposed_mean; r_prp_std(ri)=sr.rmse_proposed_std;
    r_orc_mean(ri)=sr.rmse_oracle_mean;   r_orc_std(ri)=sr.rmse_oracle_std;
    r_glb_mean(ri)=sr.rmse_global_mean;   r_glb_std(ri)=sr.rmse_global_std;
end
r_fin=r_grid(~isinf(r_grid)); r_step=r_fin(end)-r_fin(end-1);
r_plot=r_grid; r_plot(isinf(r_plot))=r_fin(end)+r_step; full_x=r_plot(end);
fig1=figure('Visible','off'); hold on; grid on;
errorbar(r_plot,r_prp_mean,r_prp_std,'b-o','CapSize',4,'MarkerSize',6);
errorbar(r_plot,r_orc_mean,r_orc_std,'g-^','CapSize',4,'MarkerSize',6);
errorbar(r_plot,r_glb_mean,r_glb_std,'r-s','CapSize',4,'MarkerSize',6);
yline(stats.rmse_avg_mean,'k--'); xline(full_x,'k:','LineWidth',1.2);
xticks([r_fin,full_x]);
xlbls=arrayfun(@num2str,r_fin,'UniformOutput',false); xlbls{end+1}='Full';
xticklabels(xlbls);
xlabel('PCR Rank r'); ylabel('RMSE (cm)'); title('RMSE vs PCR Rank');
legend({'Proposed (PCA-LDA + Dir-PCR)','Oracle Lower Bound','Global PCR','Mean Trajectory'},'Location','Best');
saveas(fig1,fullfile(out_dir,'Fig1_rmse_vs_r.png')); fprintf('Fig1 saved.\n');

%% Fig 2: Error over time
t_ax=stats.times(:)';
fig2=figure('Visible','off'); hold on; grid on;
mu=stats.err_time_mean';      sd=stats.err_time_std';
fill([t_ax fliplr(t_ax)],[mu+sd fliplr(mu-sd)],[0 0.45 0.74],'FaceAlpha',0.15,'EdgeColor','none');
plot(t_ax,mu,'-','Color',[0 0.45 0.74],'LineWidth',1.8);
mu=stats.err_time_orc_mean';  sd=stats.err_time_orc_std';
fill([t_ax fliplr(t_ax)],[mu+sd fliplr(mu-sd)],[0.17 0.63 0.17],'FaceAlpha',0.15,'EdgeColor','none');
plot(t_ax,mu,'-','Color',[0.17 0.63 0.17],'LineWidth',1.8);
mu=stats.err_time_glb_mean';  sd=stats.err_time_glb_std';
fill([t_ax fliplr(t_ax)],[mu+sd fliplr(mu-sd)],[0.84 0.15 0.16],'FaceAlpha',0.15,'EdgeColor','none');
plot(t_ax,mu,'-','Color',[0.84 0.15 0.16],'LineWidth',1.8);
mu=stats.err_time_kf_mean';   sd=stats.err_time_kf_std';
fill([t_ax fliplr(t_ax)],[mu+sd fliplr(mu-sd)],[0.93 0.69 0.13],'FaceAlpha',0.15,'EdgeColor','none');
plot(t_ax,mu,'-','Color',[0.93 0.69 0.13],'LineWidth',1.8);
xlabel('Time (ms)'); ylabel('Mean Position Error (cm)'); title('Decoding Error Over Time');
legend({'','Proposed (PCA-LDA + Dir-PCR)','','Oracle Lower Bound','','Global PCR','','Global Kalman Filter'},'Location','northwest');
xlim([min(t_ax) max(t_ax)]);
saveas(fig2,fullfile(out_dir,'Fig2_error_over_time.png')); fprintf('Fig2 saved.\n');

%% Trajectory data: one representative split
% Use a fixed seed split so figures are reproducible
rng(cfg.rng_base + 999);
n_tr = max(1, floor(size(trial,1) * cfg.sP_default)); idx = randperm(size(trial,1));
tr_rep = trial(idx(1:n_tr), :); te_rep = trial(idx(n_tr+1:end), :);
mp_rep = positionEstimatorTraining(tr_rep, struct());

num_dir = size(te_rep, 2);
times   = cfg.times_decode;
n_t     = length(times);

% Decode all test trials for direction 1 (trial 1 per direction for clean plot)
true_trajs = cell(num_dir,1);
pred_trajs = cell(num_dir,1);
for d = 1:num_dir
    td = te_rep(1, d);
    Tmax = size(td.spikes, 2);
    mp_d = mp_rep;
    mp_d.current_dir=[]; mp_d.current_class_id=0;
    mp_d.is_initialized=false; mp_d.oracle_mode=false;
    true_xy = nan(2, n_t);
    pred_xy = nan(2, n_t);
    dec = [];
    for ti = 1:n_t
        t = times(ti);
        if t > Tmax, continue; end
        ps.trialId=td.trialId; ps.spikes=td.spikes(:,1:t);
        ps.decodedHandPos=dec; ps.startHandPos=td.handPos(1:2,1);
        
        % The decoder acts like a real-time system: we feed it spikes up to time 't',
        % and it outputs the updated position. Crucially, it also returns the updated 'mp_d'
        % state, which remembers the current predicted direction for the next time step.
        [xp,yp,mp_d] = positionEstimator(ps, mp_d);
        dec = [dec,[xp;yp]]; 
        true_xy(:,ti) = td.handPos(1:2,t);
        pred_xy(:,ti) = [xp;yp];
    end
    true_trajs{d} = true_xy;
    pred_trajs{d} = pred_xy;
end

%% Fig 3: Trajectory overview (all 8 directions on one axes)
fig3 = figure('Visible','off','Position',[40,40,700,600]);
hold on; grid on; axis equal;
for d = 1:num_dir
    tx = true_trajs{d}; px = pred_trajs{d};
    ok = all(isfinite(tx),1);
    plot(tx(1,ok), tx(2,ok), '-',  'Color', cols8(d,:), 'LineWidth',1.8);
    plot(px(1,ok), px(2,ok), '--', 'Color', cols8(d,:), 'LineWidth',1.2);
    % start marker
    fi = find(ok,1);
    if ~isempty(fi)
        plot(tx(1,fi), tx(2,fi), 'o', 'MarkerFaceColor', cols8(d,:), ...
            'MarkerEdgeColor','k', 'MarkerSize',6);
    end
end
% Legend: one solid + one dashed entry only
h(1)=plot(nan,nan,'k-',  'LineWidth',1.8);
h(2)=plot(nan,nan,'k--', 'LineWidth',1.2);
h(3)=plot(nan,nan,'ko',  'MarkerFaceColor','k','MarkerSize',6);
legend(h,{'True','Predicted','Start'},'Location','best');
xlabel('x (cm)'); ylabel('y (cm)');
title('Trajectory Overview — All 8 Directions (Test Trial 1)');
saveas(fig3,fullfile(out_dir,'Fig3_trajectories_overview.png')); fprintf('Fig3 saved.\n');



%% Save bundle
save(fullfile(out_dir,'main_results_bundle.mat'), ...
    'stats','r_prp_mean','r_prp_std','r_orc_mean','r_orc_std', ...
    'r_glb_mean','r_glb_std','r_grid','r_plot','cfg');
fprintf('main_results_bundle.mat saved.\\nDone.\\n');
