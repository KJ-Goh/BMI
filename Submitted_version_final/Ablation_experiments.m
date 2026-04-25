%   Outputs -> experiment_outputs/
%     FigA1_binsize_ablation.png   dt_reg (RMSE) + dt_class (accuracy) dual-panel
%     FigA2_split_ratio.png        training data size -> RMSE: Proposed vs Upper Bound
%     FigA3_lda_ablation.png       4-condition classifier ablation
%     FigA4_mpca_sweep.png         M_pca sweep (RMSE + accuracy)
%     FigA5_mlda_sweep.png         M_lda sweep (RMSE + accuracy)
%     FigA6_dtclass_sweep.png      dt_class sweep (RMSE + accuracy)
%     FigA7_robustness.png         robustness to noise and missing data
%     FigA8_heatmap_dt_r.png       dt_reg x r_pcr heatmap
%     TableA1_binsize_ablation.csv
%     ablation_bundle.mat
clc;
clear;

mode    = 'paper';
out_dir = './experiment_outputs/';
if ~exist(out_dir,'dir'), mkdir(out_dir); end  
set(0,'DefaultAxesFontSize',11); set(0,'DefaultLineLineWidth',1.5);

cfg.sP_default   = 0.7;
cfg.rng_base     = 2013;
cfg.times_decode = 320:20:560; 
if strcmp(mode,'quick')
    cfg.num_mc_iters = 5;
    Mpca_list    = [20, 50, 100, 150];
    Mlda_list    = [1, 3, 5, 7];
    dtclass_list = [40, 80, 160];
    sP_list      = [0.5, 0.7, 0.9];
else
    cfg.num_mc_iters = 50;
    Mpca_list    = [10, 20, 50, 80, 100, 150];
    Mlda_list    = [1, 2, 3, 5, 7];
    dtclass_list = [20, 40, 80, 160];
    sP_list      = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
end
n_it = cfg.num_mc_iters;
load('monkeydata_training.mat');  


%% A1: Temporal resolution ablation
fprintf('A1: dt_reg sweep\n');
dt_reg_grid=[10,20,40,80,160]; n_dtr=length(dt_reg_grid);
a1r_mean=zeros(1,n_dtr); a1r_std=zeros(1,n_dtr);
a1r_spk=zeros(1,n_dtr);   % mean spike count per bin (noise proxy)
for di=1:n_dtr
    dt=dt_reg_grid(di); fprintf('  dt_reg=%d ms\n',dt);
    cfg_dt=cfg; cfg_dt.times_decode=320:20:560; 
    s=MC_benchmarks(trial,cfg_dt,struct('dt_reg',dt));
    a1r_mean(di)=s.rmse_proposed_mean; a1r_std(di)=s.rmse_proposed_std;
    % mean spikes per bin across all neurons/trials/directions at t=560
    t_ref=560; nb=floor(t_ref/dt); nn=size(trial(1,1).spikes,1);
    spk_sum=0; spk_n=0;
    for d=1:size(trial,2)
        for i=1:size(trial,1)
            sp=trial(i,d).spikes(:,1:min(t_ref,size(trial(i,d).spikes,2)));
            for b=1:nb
                cols=(b-1)*dt+1:min(b*dt,size(sp,2));
                spk_sum=spk_sum+sum(sum(sp(:,cols))); spk_n=spk_n+nn;
            end
        end
    end
    a1r_spk(di)=spk_sum/max(spk_n,1);
end

fprintf('A1: dt_class sweep\n');
dt_cls_grid=[20,40,80,160]; 
[a1c_acc_mean, a1c_acc_std, ~, ~] = sweep_param(trial, cfg, dt_cls_grid, 'dt_class', 'dt_class');

figA1=figure('Visible','off','Position',[40,40,1000,420]);
% Left panel: RMSE + mean spikes/bin (dual axis)
subplot(1,2,1);
yyaxis left
bar(1:n_dtr,a1r_mean,0.5,'FaceColor','flat','EdgeColor','k', ...
    'CData',[[0.6 0.6 0.6];[0.2 0.5 0.8];[0.6 0.6 0.6];[0.6 0.6 0.6];[0.6 0.6 0.6]]);
hold on;
errorbar(1:n_dtr,a1r_mean,a1r_std,'k.','LineWidth',1.5,'CapSize',8);
ylabel('RMSE (cm)');
yyaxis right
plot(1:n_dtr,a1r_spk,'-^','Color',[0.84 0.15 0.16],'MarkerFaceColor',[0.84 0.15 0.16],'LineWidth',1.5,'MarkerSize',8);
ylabel('Mean Spikes per Bin (noise proxy)');
xticks(1:n_dtr); xticklabels({'10 ms','20 ms','40 ms','80 ms','160 ms'});
xlabel('Regression Bin Width (\Deltat_{reg})'); grid on;
title('(a) Regression: RMSE and Spike Density vs. Bin Width');
legend({'RMSE (mean \pm std)','','Mean spikes/bin'},'Location','best');

% Right panel: accuracy vs dt_class — y-axis zoomed to show differences
subplot(1,2,2); hold on; grid on;
n_dtc=length(dt_cls_grid);
bh2=bar(1:n_dtc,a1c_acc_mean,0.5,'FaceColor','flat','EdgeColor','k');
bh2.CData=repmat([0.6 0.6 0.6],n_dtc,1); bh2.CData(3,:)=[0.84 0.15 0.16];
errorbar(1:n_dtc,a1c_acc_mean,a1c_acc_std,'k.','LineWidth',1.5,'CapSize',8);
xticks(1:n_dtc); xticklabels({'20 ms','40 ms','80 ms','160 ms'});
xlabel('Classification Bin Width (\Deltat_{class})'); ylabel('Classification Accuracy');
title('(b) Classification: Accuracy vs. Bin Width');
acc_lo = max(0, min(a1c_acc_mean - a1c_acc_std) - 0.05);
acc_hi = min(1, max(a1c_acc_mean + a1c_acc_std) + 0.05);
ylim([acc_lo acc_hi]);

sgtitle('Temporal Resolution vs. Decoding Performance');
saveas(figA1,fullfile(out_dir,'FigA1_binsize_ablation.png')); fprintf('FigA1 saved.\n');
writetable(table(dt_reg_grid',a1r_mean',a1r_std','VariableNames',{'BinSize_ms','Mean_RMSE_cm','Std_RMSE_cm'}), ...
    fullfile(out_dir,'TableA1_binsize_ablation.csv'));

%% A2: Training data size effect
fprintf('A2: Training data size sweep\n');
ns=length(sP_list);
a2_avg_mean=zeros(ns,1); a2_avg_std=zeros(ns,1);
a2_glb_mean=zeros(ns,1); a2_glb_std=zeros(ns,1);
a2_prp_mean=zeros(ns,1); a2_prp_std=zeros(ns,1);
a2_orc_mean=zeros(ns,1); a2_orc_std=zeros(ns,1);
for is=1:ns
    sP=sP_list(is); fprintf('  sP=%.1f\n',sP);
    cfg_sp=cfg; cfg_sp.sP_default=sP;
    s=MC_benchmarks(trial,cfg_sp);
    a2_avg_mean(is)=s.rmse_avg_mean;      a2_avg_std(is)=s.rmse_avg_std;
    a2_glb_mean(is)=s.rmse_global_mean;   a2_glb_std(is)=s.rmse_global_std;
    a2_prp_mean(is)=s.rmse_proposed_mean; a2_prp_std(is)=s.rmse_proposed_std;
    a2_orc_mean(is)=s.rmse_oracle_mean;   a2_orc_std(is)=s.rmse_oracle_std;
end
figA2=figure('Visible','off','Position',[40,40,650,420]); hold on; grid on;
errorbar(sP_list,a2_avg_mean,a2_avg_std,'--d','Color',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5],'CapSize',4,'LineWidth',1.2);
errorbar(sP_list,a2_glb_mean,a2_glb_std,'--s','Color',[0.84 0.15 0.16],'MarkerFaceColor',[0.84 0.15 0.16],'CapSize',4,'LineWidth',1.2);
errorbar(sP_list,a2_prp_mean,a2_prp_std,'-o','Color',[0.2 0.5 0.8],'MarkerFaceColor',[0.2 0.5 0.8],'CapSize',4,'LineWidth',1.8);
errorbar(sP_list,a2_orc_mean,a2_orc_std,'-^','Color',[0.2 0.7 0.3],'MarkerFaceColor',[0.2 0.7 0.3],'CapSize',4,'LineWidth',1.8);
xlabel('Training Fraction (s_P)'); ylabel('RMSE (cm)');
title('Effect of Training Data Size on Decoding Performance');
legend({'Mean Trajectory','Global PCR','Proposed (PCA-LDA + Dir-PCR)','Oracle Lower Bound'},'Location','best');
saveas(figA2,fullfile(out_dir,'FigA2_split_ratio.png')); fprintf('FigA2 saved.\n');

%% A3: Classifier ablation
fprintf('A3: Classifier ablation\n');
acc_cond=zeros(6,n_it); rmse_cond=zeros(6,n_it);
for it=1:n_it
    seed=cfg.rng_base+it; rng(seed);
    [tr,te]=helper_split_trials(trial,cfg.sP_default,seed);
    num_dir=size(te,2); num_te=size(te,1);
    mp_lda=positionEstimatorTraining(tr,struct('use_lda',true));
    mp_pca=positionEstimatorTraining(tr,struct('use_lda',false));
    class_times = mp_lda.class_times;
    dt_class = mp_lda.dt_class;
    M_pca_cls = size(mp_pca.classifier(1).Wopt, 2);

    knn_bank = train_optional_classifier_bank(tr, class_times, dt_class, M_pca_cls, 'knn');
    svm_bank = train_optional_classifier_bank(tr, class_times, dt_class, M_pca_cls, 'svm_linear');

    acc_cond(1,it)=1/num_dir;
    acc_cond(2,it)=evaluate_classifier_accuracy(te, mp_pca.classifier, 'centroid');
    acc_cond(3,it)=evaluate_classifier_accuracy(te, mp_lda.classifier, 'centroid');
    acc_cond(4,it)=evaluate_classifier_accuracy(te, knn_bank, 'model');
    acc_cond(5,it)=evaluate_classifier_accuracy(te, svm_bank, 'model');
    acc_cond(6,it)=1.0;

    mean_traj = compute_mean_trajectory(tr, cfg.times_decode);
    rmse_cond(1,it)=evaluate_chance_rmse(te, mean_traj, cfg.times_decode);
    rmse_cond(2,it)=evaluate_classifier_rmse(te, mp_lda, mp_pca.classifier, class_times, cfg.times_decode, 'centroid');
    rmse_cond(3,it)=evaluate_classifier_rmse(te, mp_lda, mp_lda.classifier, class_times, cfg.times_decode, 'centroid');
    rmse_cond(4,it)=evaluate_classifier_rmse(te, mp_lda, knn_bank, class_times, cfg.times_decode, 'model');
    rmse_cond(5,it)=evaluate_classifier_rmse(te, mp_lda, svm_bank, class_times, cfg.times_decode, 'model');
    rmse_cond(6,it)=evaluate_classifier_rmse(te, mp_lda, [], class_times, cfg.times_decode, 'oracle');
end
cond_labels={'Chance','PCA','PCA-LDA','PCA-KNN','PCA-SVM','Oracle Lower Bound'};
cond_colors=[0.6 0.6 0.6; 0.9 0.6 0.2; 0.2 0.5 0.8; 0.55 0.35 0.75; 0.15 0.65 0.65; 0.2 0.7 0.3];
acc_mu=mean(acc_cond,2); acc_sd=std(acc_cond,0,2);
rmse_mu=mean(rmse_cond,2); rmse_sd=std(rmse_cond,0,2);
figA3=figure('Visible','off','Position',[40,40,1000,420]);
subplot(1,2,1); hold on; grid on;
for c=1:numel(cond_labels)
    bar(c,acc_mu(c),0.5,'FaceColor',cond_colors(c,:),'EdgeColor','k');
    errorbar(c,acc_mu(c),acc_sd(c),'k.','CapSize',8,'LineWidth',1.5);
    text(c,acc_mu(c)+acc_sd(c)+0.02,sprintf('%.2f',acc_mu(c)),'HorizontalAlignment','center','FontSize',9);
end
xticks(1:numel(cond_labels)); xticklabels(cond_labels); ylabel('Classification Accuracy');
title('(a) Classification Accuracy'); ylim([0 1.15]);
subplot(1,2,2); hold on; grid on;
for c=1:numel(cond_labels)
    bar(c,rmse_mu(c),0.5,'FaceColor',cond_colors(c,:),'EdgeColor','k');
    errorbar(c,rmse_mu(c),rmse_sd(c),'k.','CapSize',8,'LineWidth',1.5);
    text(c,rmse_mu(c)+rmse_sd(c)+0.01*max(rmse_mu),sprintf('%.3f',rmse_mu(c)),'HorizontalAlignment','center','FontSize',9);
end
xticks(1:numel(cond_labels)); xticklabels(cond_labels); ylabel('RMSE (cm)');
title('(b) Trajectory RMSE');
sgtitle('Classifier Comparison: Unified Direction Classifier Baselines');
saveas(figA3,fullfile(out_dir,'FigA3_lda_ablation.png')); fprintf('FigA3 saved.\n');

%% A4-A6: Parameter sweeps (M_pca / M_lda / dt_class)
clr_rmse=[0.2 0.5 0.8]; clr_acc=[0.84 0.15 0.16];

fprintf('Sweeping M_pca, M_lda, dt_class\n');
[sw_mpca_acc_mean,sw_mpca_acc_std,sw_mpca_rmse_mean,sw_mpca_rmse_std] = ...
    sweep_param(trial, cfg, Mpca_list,    'M_pca',    'M_pca');
[sw_mlda_acc_mean,sw_mlda_acc_std,sw_mlda_rmse_mean,sw_mlda_rmse_std] = ...
    sweep_param(trial, cfg, Mlda_list,    'M_lda',    'M_lda');
[sw_dtc_acc_mean,sw_dtc_acc_std,sw_dtc_rmse_mean,sw_dtc_rmse_std] = ...
    sweep_param(trial, cfg, dtclass_list, 'dt_class', 'dt_class');

fprintf('A7: SNR robustness sweep\n');
snr_db_list = [5, 10, 20, 30, Inf]; n_snr = length(snr_db_list);
[rob_prp_mean, rob_prp_std, rob_glb_mean, rob_glb_std]=deal(zeros(n_snr,1));
for is = 1:n_snr
    snr_db = snr_db_list(is);
    fprintf('  SNR=%s dB\n', num2str(snr_db));
    rp = zeros(n_it,1);rg = zeros(n_it,1);
    for it = 1:n_it
        seed = cfg.rng_base + it;
        rng(seed);
        [tr, te] = helper_split_trials(trial, cfg.sP_default, seed);
        te_noisy = te;
        if ~isinf(snr_db)
            for d = 1:size(te,2)
                for i = 1:size(te,1)
                    sp = te(i,d).spikes;
                    sig_pwr = mean(sp(:).^2) + eps;
                    noise_std = sqrt(sig_pwr / (10^(snr_db/10)));
                    te_noisy(i,d).spikes = max(0, sp + noise_std * randn(size(sp)));
                end
            end
        end
        mp   = positionEstimatorTraining(tr, struct());
        mp_g = positionEstimatorTraining(tr, struct('global_decoder', true));
        sq_p = 0;sq_g = 0;
        np   = 0;
        for d = 1:size(te_noisy,2)
            for i = 1:size(te_noisy,1)
                td = te_noisy(i,d);
                mp_r = mp;
                mp_r.current_dir = [];
                mp_r.current_class_id = 0;
                mp_r.is_initialized = false;
                mp_r.oracle_mode = false;
                mp_gr = mp_g;
                mp_gr.current_dir = [];
                mp_gr.current_class_id = 0;
                mp_gr.is_initialized = false;
                mp_gr.oracle_mode = false;
                dp = [];dg = [];

                for t2 = cfg.times_decode
                    if t2 > size(td.spikes,2)
                        continue;
                    end
                    ps = struct();
                    ps.trialId = td.trialId;
                    ps.spikes = td.spikes(:,1:t2);
                    ps.decodedHandPos = dp;
                    ps.startHandPos = td.handPos(1:2,1);
                    pg = ps; pg.decodedHandPos = dg;
                    [xp, yp, mp_r]  = positionEstimator(ps, mp_r);
                    [xg, yg, mp_gr] = positionEstimator(pg, mp_gr);
                    dp = [dp, [xp; yp]];
                    dg = [dg, [xg; yg]];
                    xy = td.handPos(1:2,t2);
                    sq_p = sq_p + (xp - xy(1))^2 + (yp - xy(2))^2;
                    sq_g = sq_g + (xg - xy(1))^2 + (yg - xy(2))^2;
                    np = np + 1;
                end
            end
        end
        rp(it) = sqrt(sq_p / max(np,1));
        rg(it) = sqrt(sq_g / max(np,1));
    end

    rob_prp_mean(is) = mean(rp);
    rob_prp_std(is)  = std(rp);

    rob_glb_mean(is) = mean(rg);
    rob_glb_std(is)  = std(rg);

end
fprintf('A8: Missing data robustness sweep\n');
miss_list = [0, 0.1, 0.2, 0.3];
n_miss = length(miss_list);
miss_prp_mean = zeros(n_miss,1); miss_prp_std = zeros(n_miss,1);
miss_glb_mean = zeros(n_miss,1); miss_glb_std = zeros(n_miss,1);
for im2 = 1:n_miss
    mr2 = miss_list(im2); fprintf('  missing=%.0f%%\n', mr2*100);
    rp = zeros(n_it,1); rg = zeros(n_it,1);
    for it = 1:n_it
        seed = cfg.rng_base+it; rng(seed);
        [tr,te] = helper_split_trials(trial, cfg.sP_default, seed);
        te_miss = te;
        if mr2 > 0
            for d=1:size(te,2), for i=1:size(te,1)
                sp=te(i,d).spikes; te_miss(i,d).spikes=sp.*(rand(size(sp))>=mr2);
            end, end
        end
        mp=positionEstimatorTraining(tr,struct()); mp_g=positionEstimatorTraining(tr,struct('global_decoder',true));
        sq_p=0; sq_g=0; np=0;
        for d=1:size(te_miss,2), for i=1:size(te_miss,1)
            td=te_miss(i,d); mp_r=mp; mp_r.current_dir=[]; mp_r.current_class_id=0; mp_r.is_initialized=false; mp_r.oracle_mode=false;
            mp_gr=mp_g; mp_gr.current_dir=[]; mp_gr.current_class_id=0; mp_gr.is_initialized=false; mp_gr.oracle_mode=false;
            dp=[]; dg=[];
            for t2=cfg.times_decode
                if t2>size(td.spikes,2), continue; end
                ps.trialId=td.trialId; ps.spikes=td.spikes(:,1:t2); ps.decodedHandPos=dp; ps.startHandPos=td.handPos(1:2,1);
                pg=ps; pg.decodedHandPos=dg;
                [xp,yp,mp_r]=positionEstimator(ps,mp_r); [xg,yg,mp_gr]=positionEstimator(pg,mp_gr);
                dp=[dp,[xp;yp]]; dg=[dg,[xg;yg]]; xy=td.handPos(1:2,t2);
                sq_p=sq_p+(xp-xy(1))^2+(yp-xy(2))^2; sq_g=sq_g+(xg-xy(1))^2+(yg-xy(2))^2; np=np+1;
            end
        end, end
        rp(it)=sqrt(sq_p/max(np,1)); rg(it)=sqrt(sq_g/max(np,1));
    end
    miss_prp_mean(im2)=mean(rp); miss_prp_std(im2)=std(rp);
    miss_glb_mean(im2)=mean(rg); miss_glb_std(im2)=std(rg);
end

figRob = figure('Visible','off','Position',[40,40,1000,420]);
subplot(1,2,1); hold on; grid on;
errorbar(1:n_snr,rob_prp_mean,rob_prp_std,'-o','Color',[0.2 0.5 0.8],'MarkerFaceColor',[0.2 0.5 0.8],'CapSize',4,'LineWidth',1.5);
errorbar(1:n_snr,rob_glb_mean,rob_glb_std,'-s','Color',[0.84 0.15 0.16],'MarkerFaceColor',[0.84 0.15 0.16],'CapSize',4,'LineWidth',1.5);
xticks(1:n_snr); xticklabels({'5 dB','10 dB','20 dB','30 dB','Clean'});
xlabel('Input SNR'); ylabel('RMSE (cm)'); title('(a) Gaussian Noise Robustness');
legend({'Proposed (PCA-LDA + Dir-PCR)','Global PCR'},'Location','best');
subplot(1,2,2); hold on; grid on;
errorbar(miss_list*100,miss_prp_mean,miss_prp_std,'-o','Color',[0.2 0.5 0.8],'MarkerFaceColor',[0.2 0.5 0.8],'CapSize',4,'LineWidth',1.5);
errorbar(miss_list*100,miss_glb_mean,miss_glb_std,'-s','Color',[0.84 0.15 0.16],'MarkerFaceColor',[0.84 0.15 0.16],'CapSize',4,'LineWidth',1.5);
xlabel('Missing Spike Rate (%)'); ylabel('RMSE (cm)'); title('(b) Missing Data Robustness');
legend({'Proposed (PCA-LDA + Dir-PCR)','Global PCR'},'Location','best');
sgtitle('Robustness to Noise and Missing Data');
saveas(figRob, fullfile(out_dir,'FigA7_robustness.png')); fprintf('FigA7 saved.\n');

%% A9: Hyperparameter coupling heatmap
fprintf('A9: dt_reg x r_pcr heatmap\n');
if strcmp(mode,'quick'), hm_dt=[10,20,40]; hm_r=[5,20,80];
else, hm_dt=[10,20,40,80]; hm_r=[5,10,20,40,80,Inf]; end
n_hdt=length(hm_dt); n_hr=length(hm_r);
hm_rmse=zeros(n_hr,n_hdt);
for ir=1:n_hr
    for idt=1:n_hdt
        dt=hm_dt(idt); rv=hm_r(ir);
        if isinf(rv), rp_val=[]; else, rp_val=rv; end
        fprintf('  dt=%d r=%s\n',dt,num2str(rv));
        cfg_hm=cfg; t_all=dt:dt:560; cfg_hm.times_decode=t_all(t_all>=320);  
        s_hm=MC_benchmarks(trial,cfg_hm,struct('dt_reg',dt,'r_pcr',rp_val));
        hm_rmse(ir,idt)=s_hm.rmse_proposed_mean;
    end
end
figHM=figure('Visible','off','Position',[40,40,600,480]);
imagesc(hm_rmse); colorbar; colormap(flipud(hot));
xticks(1:n_hdt); xticklabels(arrayfun(@(x) sprintf('%d ms',x),hm_dt,'UniformOutput',false));
ytlbls=arrayfun(@num2str,hm_r,'UniformOutput',false); ytlbls{end}='Full';
yticks(1:n_hr); yticklabels(ytlbls);
xlabel('Regression Bin Width (\Deltat_{reg})'); ylabel('PCR Rank (r)');
title('Hyperparameter Interaction: Mean RMSE (cm)');
for ir=1:n_hr
    for idt=1:n_hdt
        text(idt,ir,sprintf('%.3f',hm_rmse(ir,idt)),'HorizontalAlignment','center','FontSize',8,'Color','w','FontWeight','bold');
    end
end
saveas(figHM, fullfile(out_dir,'FigA8_heatmap_dt_r.png')); fprintf('FigA8 saved.\n');

%% Save extended bundle
save(fullfile(out_dir,'ablation_bundle.mat'), ...
    'a1r_mean','a1r_std','a1r_spk','dt_reg_grid','a1c_acc_mean','a1c_acc_std','dt_cls_grid', ...
    'a2_avg_mean','a2_avg_std','a2_glb_mean','a2_glb_std', ...
    'a2_prp_mean','a2_prp_std','a2_orc_mean','a2_orc_std','sP_list', ...
    'acc_cond','rmse_cond', ...
    'sw_mpca_acc_mean','sw_mpca_acc_std','sw_mpca_rmse_mean','sw_mpca_rmse_std','Mpca_list', ...
    'sw_mlda_acc_mean','sw_mlda_acc_std','sw_mlda_rmse_mean','sw_mlda_rmse_std','Mlda_list', ...
    'sw_dtc_acc_mean','sw_dtc_acc_std','sw_dtc_rmse_mean','sw_dtc_rmse_std','dtclass_list', ...
    'rob_prp_mean','rob_prp_std','rob_glb_mean','rob_glb_std','snr_db_list', ...
    'miss_prp_mean','miss_prp_std','miss_glb_mean','miss_glb_std','miss_list', ...
    'hm_rmse','hm_dt','hm_r','cfg');
fprintf('ablation_bundle.mat (extended) saved.\nDone.\n');

%% Local helper functions
function [am,as,rm,rs] = sweep_param(trial, cfg, vlist, pname, plbl)
n=length(vlist); [am,as,rm,rs]=deal(zeros(n,1));
for iv=1:n
    fprintf('  %s=%g\\n', plbl, vlist(iv)); av=zeros(cfg.num_mc_iters,1); rv=zeros(cfg.num_mc_iters,1);
    for it=1:cfg.num_mc_iters
        rng(cfg.rng_base+it); idx=randperm(size(trial,1)); n_tr=max(1,floor(length(idx)*cfg.sP_default));
        tr=trial(idx(1:n_tr),:); te=trial(idx(n_tr+1:end),:);
        mp=positionEstimatorTraining(tr,struct(pname,vlist(iv)));
        cor=0; ntot=0; sq=0; np=0;
        for ci=1:length(mp.classifier)
            clf=mp.classifier(ci); dt_c=clf.dt; tc=clf.t; nb=floor(tc/dt_c); nn=size(te(1,1).spikes,1);
            for d=1:size(te,2)
                for i=1:size(te,1)
                    sp=te(i,d).spikes(:,1:min(tc,size(te(i,d).spikes,2))); f=zeros(nn*nb,1);
                    for b=1:nb, f((b-1)*nn+1:b*nn)=sum(sp(:,(b-1)*dt_c+1:b*dt_c),2); end
                    [~,prd]=min(sum((clf.Wopt'*(f-clf.mx)-clf.centroids).^2,1));
                    if prd==d, cor=cor+1; end; ntot=ntot+1;
                end
            end
        end
        for d=1:size(te,2)
            for i=1:size(te,1)
                td=te(i,d); mr=mp; mr.current_dir=[]; mr.current_class_id=0; mr.is_initialized=false; mr.oracle_mode=false; dec=[];
                for t2=cfg.times_decode
                    if t2>size(td.spikes,2), continue; end
                    ps.trialId=td.trialId; ps.spikes=td.spikes(:,1:t2); ps.decodedHandPos=dec; ps.startHandPos=td.handPos(1:2,1);
                    [xp,yp,mr]=positionEstimator(ps,mr); dec=[dec,[xp;yp]]; 
                    sq=sq+(xp-td.handPos(1,t2))^2+(yp-td.handPos(2,t2))^2; np=np+1;
                end
            end
        end
        av(it)=cor/max(1,ntot); rv(it)=sqrt(sq/max(np,1));
    end
    am(iv)=mean(av); as(iv)=std(av); rm(iv)=mean(rv); rs(iv)=std(rv);
end
end

function mt = compute_mean_trajectory(tr, tms)
mt=zeros(size(tr,2),2,length(tms));
for d=1:size(tr,2), for ti=1:length(tms), xy=zeros(size(tr,1),2);
    for i=1:size(tr,1), xy(i,:)=tr(i,d).handPos(1:2,min(tms(ti),size(tr(i,d).handPos,2)))'; end
    mt(d,:,ti)=mean(xy,1);
end, end
end

function rmse = evaluate_chance_rmse(te, mt, tms)
sq=0; np=0;
for d=1:size(te,2), for i=1:size(te,1), for ti=1:length(tms)
    t=tms(ti); if t>size(te(i,d).handPos,2), continue; end
    dr=randi([1,size(te,2)]); xy=mt(dr,:,ti)+(te(i,d).handPos(1:2,1)'-mt(dr,:,1));
    sq=sq+sum((te(i,d).handPos(1:2,t)'-xy).^2); np=np+1;
end, end, end
rmse=sqrt(sq/max(np,1));
end

function b = train_optional_classifier_bank(tr, ct, dt, M_pca, md)
b=repmat(struct('method',md,'t',[],'dt',dt,'mx',[],'Wpca',[],'mdl',[]),length(ct),1);
for ci=1:length(ct)
    t=ct(ci); nb=floor(t/dt); X=[]; Y=[];
    for d=1:size(tr,2), for i=1:size(tr,1)
        sp=tr(i,d).spikes(:,1:t); f=zeros(1,size(sp,1)*nb);
        for bb=1:nb, f((bb-1)*size(sp,1)+1:bb*size(sp,1))=sum(sp(:,(bb-1)*dt+1:bb*dt),2)'; end
        X=[X; f]; Y=[Y; d]; 
    end, end
    mx=mean(X,1); [~,~,V]=svd(X-mx,'econ'); W=V(:,1:min(M_pca,size(V,2))); Z=(X-mx)*W;
    if strcmp(md,'knn'), mdl=fitcknn(Z,Y,'NumNeighbors',5); else, mdl=fitcecoc(Z,Y,'Learners',templateSVM('KernelFunction','linear')); end
    b(ci).t=t; b(ci).mx=mx; b(ci).Wpca=W; b(ci).mdl=mdl;
end
end

function acc = evaluate_classifier_accuracy(te, bnk, md)
cor=0; ntot=0; nn=size(te(1,1).spikes,1);
for ci=1:length(bnk)
    t=bnk(ci).t; dt=bnk(ci).dt; nb=floor(t/dt); 
    for d=1:size(te,2), for i=1:size(te,1)
        sp=te(i,d).spikes(:,1:min(t,size(te(i,d).spikes,2))); f=zeros(1,nn*nb);
        for bb=1:nb, f((bb-1)*nn+1:bb*nn)=sum(sp(:,(bb-1)*dt+1:bb*dt),2)'; end
        if strcmp(md,'centroid'), [~,prd]=min(sum((bnk(ci).Wopt'*(f'-bnk(ci).mx)-bnk(ci).centroids).^2,1));
        else, prd=predict(bnk(ci).mdl,(f-bnk(ci).mx)*bnk(ci).Wpca); prd=prd(1); end
        if prd==d, cor=cor+1; end; ntot=ntot+1;
    end, end
end
acc=cor/max(ntot,1);
end

function rmse = evaluate_classifier_rmse(te, mp, bnk, ct, tms, md)
sq=0; np=0; nn=size(te(1,1).spikes,1);
for d=1:size(te,2), for i=1:size(te,1)
    td=te(i,d); 
    for ti=1:length(tms)
        t=tms(ti); if t>size(td.spikes,2), continue; end
        cid=find(ct==t,1);
        if strcmp(md,'oracle'), prd=d; 
        elseif ~isempty(cid)
            clf=bnk(cid); tc=clf.t; dt_c=clf.dt; nb=floor(tc/dt_c); f=zeros(1,nn*nb);
            for bb=1:nb, f((bb-1)*nn+1:bb*nn)=sum(td.spikes(:,(bb-1)*dt_c+1:bb*dt_c),2)'; end
            if strcmp(md,'centroid'), [~,prd]=min(sum((clf.Wopt'*(f'-clf.mx)-clf.centroids).^2,1));
            else, prd=predict(clf.mdl,(f-clf.mx)*clf.Wpca); prd=prd(1); end
        else, continue; end
        ex=mp.expert(prd).time(find(mp.reg_times<=t,1,'last')); dt_r=mp.dt_reg; nb_r=floor(ex.t/dt_r); f2=zeros(1,nn*nb_r);
        for bb=1:nb_r, f2((bb-1)*nn+1:bb*nn)=sum(td.spikes(:,(bb-1)*dt_r+1:bb*dt_r),2)'; end
        xy=(f2-ex.mx)*ex.Beta+ex.my+(td.handPos(1:2,1)'-mp.mean_starts(prd,:)); sq=sq+sum((xy'-td.handPos(1:2,t)).^2); np=np+1;
    end
end, end
rmse=sqrt(sq/max(np,1));
end

function [tr, te] = helper_split_trials(trial, sP, seed)
if nargin >= 3 && ~isempty(seed), rng(seed); end
n=size(trial,1); n_tr=max(1,floor(n*sP)); idx=randperm(n);
tr=trial(idx(1:n_tr),:); te=trial(idx(n_tr+1:end),:);
end
