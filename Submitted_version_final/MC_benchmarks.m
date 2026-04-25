function stats = MC_benchmarks(trial, cfg, opts)
if nargin < 3, opts = struct(); end
tms = cfg.times_decode(:); nt = length(tms); iters = cfg.num_mc_iters;
[r_avg, r_glb, r_prp, r_orc, r_kf, t_trn, t_inf, n_prd, t_tot] = deal(zeros(iters,1));
[e_prp, e_orc, e_glb, e_kf] = deal(zeros(nt, iters));

for it = 1:iters
    rng(cfg.rng_base + it);
    idx = randperm(size(trial,1)); n_tr = max(1, floor(length(idx) * cfg.sP_default));
    tr = trial(idx(1:n_tr), :); te = trial(idx(n_tr+1:end), :);
    [n_tr_tr, ndir] = size(tr); n_te_tr = size(te,1);
    
    % Mean Trajectory
    mtraj = zeros(ndir, 2, nt);
    for d = 1:ndir
        for ti = 1:nt
            xy = zeros(n_tr_tr, 2);
            for i = 1:n_tr_tr, xy(i,:) = tr(i,d).handPos(1:2, min(tms(ti), size(tr(i,d).handPos,2)))'; end
            mtraj(d,:,ti) = mean(xy, 1);
        end
    end

    % Kalman Filter Baseline
    % We train a standard KF here just for comparison.
    % State: [x, y, vx, vy]. Observation: [98 neurons' spike counts in 20ms bins].
    min_b = round(tms(1)/20);
    Xa=[]; Ya=[]; Xp=[]; Xn=[];
    for n=1:numel(tr)
        p=double(tr(n).handPos(1:2,20:20:end)); y=zeros(98,size(p,2));
        for b=1:size(p,2), y(:,b)=sum(tr(n).spikes(:,(b-1)*20+1:b*20),2); end
        v=[0 0; diff(p')/0.020]; x=[p-p(:,1); v'];
        Xa=[Xa x]; Ya=[Ya y]; Xp=[Xp x(:,1:end-1)]; Xn=[Xn x(:,2:end)];
    end
    % A, H, W, Q are the standard KF transition/observation matrices and noise covariances.
    % We add a tiny 1e-6 eye matrix to the covariances to avoid singular matrices later.
    A=Xn*pinv(Xp); H=Ya*pinv(Xa); R=Xn-A*Xp; W=(R*R')/max(size(R,2)-1,1); W=(W+W')/2+1e-6*eye(4);
    R=Ya-H*Xa; Q=(R*R')/max(size(R,2)-1,1); Q=(Q+Q')/2+1e-6*eye(98);
    
    se_kf=0; n_kf=0; ekf_t=zeros(nt,1); nkf_t=zeros(nt,1);
    for n=1:numel(te)
        p=double(te(n).handPos(1:2,20:20:end)); p0=p(:,1); y=zeros(98,size(p,2));
        for b=1:size(p,2), y(:,b)=sum(te(n).spikes(:,(b-1)*20+1:b*20),2); end
        x=zeros(4,1); P=100*eye(4); I=eye(4);
        for b=1:min(28,size(y,2))
            xp=A*x; Pp=A*P*A'+W; Pp=(Pp+Pp')/2; S=H*Pp*H'+Q; S=(S+S')/2; K=(Pp*H')/S;
            x=xp+K*(y(:,b)-H*xp); P=(I-K*H)*Pp*(I-K*H)'+K*Q*K'; P=(P+P')/2;
            if b>=min_b, e=norm(x(1:2)+p0-p(:,b)); se_kf=se_kf+e^2; n_kf=n_kf+1; ekf_t(b-min_b+1)=ekf_t(b-min_b+1)+e; nkf_t(b-min_b+1)=nkf_t(b-min_b+1)+1; end
        end
    end
    r_kf(it)=sqrt(se_kf/max(n_kf,1)); e_kf(nkf_t>0,it) = ekf_t(nkf_t>0) ./ nkf_t(nkf_t>0);

    % Main Decoders
    opts_g=opts; opts_g.global_decoder=true; mp_g=positionEstimatorTraining(tr, opts_g);
    mp=positionEstimatorTraining(tr, opts); t_trn(it)=mp.train_time_sec;

    % Combined Eval
    [sq_p, sq_o, sq_g, sq_a, np, na] = deal(0);
    etp=zeros(nt,1); eto=zeros(nt,1); etg=zeros(nt,1); npt=zeros(nt,1);
    inv_t=zeros(1, n_te_tr*ndir*nt); idx_t=0;

    for d=1:ndir
        for i=1:n_te_tr
            td=te(i,d); Tmax=size(td.spikes,2);
            mr=mp; mr.current_dir=[]; mr.current_class_id=0; mr.is_initialized=false; mr.oracle_mode=false;
            mo=mp; mo.current_dir=[]; mo.current_class_id=0; mo.is_initialized=false; mo.oracle_mode=true; mo.true_direction=d;
            mg=mp_g; mg.current_dir=[]; mg.current_class_id=0; mg.is_initialized=false; mg.oracle_mode=false;
            dp=[]; do=[]; dg=[];
            
            % Now simulate real-time feed: pass data up to time 't', get the output, repeat.
            % We time the proposed model (mr) using tic/toc to evaluate inference speed.
            for ti=1:nt
                t=tms(ti); if t>Tmax, continue; end
                ps.trialId=td.trialId; ps.spikes=td.spikes(:,1:t); ps.startHandPos=td.handPos(1:2,1);
                ps.decodedHandPos=dp; tic_t=tic; [xp, yp, mr]=positionEstimator(ps, mr); inv_t(idx_t+1)=toc(tic_t); idx_t=idx_t+1;
                ps.decodedHandPos=do; [xo, yo, mo]=positionEstimator(ps, mo);
                ps.decodedHandPos=dg; [xg, yg, mg]=positionEstimator(ps, mg);
                dp=[dp,[xp;yp]]; do=[do,[xo;yo]]; dg=[dg,[xg;yg]]; xy=td.handPos(1:2, t);
                ep=(xp-xy(1))^2+(yp-xy(2))^2; eo=(xo-xy(1))^2+(yo-xy(2))^2; eg=(xg-xy(1))^2+(yg-xy(2))^2;
                sq_p=sq_p+ep; sq_o=sq_o+eo; sq_g=sq_g+eg; np=np+1;
                etp(ti)=etp(ti)+sqrt(ep); eto(ti)=eto(ti)+sqrt(eo); etg(ti)=etg(ti)+sqrt(eg); npt(ti)=npt(ti)+1;
                pd=mr.current_dir; if isempty(pd), pd=1; end
                pm = mtraj(pd,:,ti)' + (td.handPos(1:2,1) - mtraj(pd,:,1)');
                sq_a=sq_a+sum((xy-pm).^2); na=na+1;
            end
        end
    end
    r_prp(it)=sqrt(sq_p/max(np,1)); r_orc(it)=sqrt(sq_o/max(np,1)); r_glb(it)=sqrt(sq_g/max(np,1)); r_avg(it)=sqrt(sq_a/max(na,1));
    t_inf(it)=mean(inv_t(1:idx_t)); t_tot(it)=sum(inv_t(1:idx_t)); n_prd(it)=np;
    for ti=1:nt, if npt(ti)>0, etp(ti)=etp(ti)/npt(ti); eto(ti)=eto(ti)/npt(ti); etg(ti)=etg(ti)/npt(ti); end; end
    e_prp(:,it)=etp; e_orc(:,it)=eto; e_glb(:,it)=etg;
end

stats = struct('rmse_avg_mean',mean(r_avg), 'rmse_avg_std',std(r_avg), ...
    'rmse_global_mean',mean(r_glb), 'rmse_global_std',std(r_glb), ...
    'rmse_proposed_mean',mean(r_prp), 'rmse_proposed_std',std(r_prp), ...
    'rmse_oracle_mean',mean(r_orc), 'rmse_oracle_std',std(r_orc), ...
    'rmse_kalman_mean',mean(r_kf), 'rmse_kalman_std',std(r_kf), ...
    'train_time_mean',mean(t_trn), 'train_time_std',std(t_trn), ...
    'inference_step_mean_sec',mean(t_inf), 'inference_step_std_sec',std(t_inf), ...
    'mean_n_predictions',mean(n_prd), 'total_inference_sec_mean',mean(t_tot), ...
    'err_time_mean',mean(e_prp,2), 'err_time_std',std(e_prp,0,2), ...
    'err_time_glb_mean',mean(e_glb,2), 'err_time_glb_std',std(e_glb,0,2), ...
    'err_time_kf_mean',mean(e_kf,2), 'err_time_kf_std',std(e_kf,0,2), ...
    'err_time_orc_mean',mean(e_orc,2), 'err_time_orc_std',std(e_orc,0,2), ...
    'times',tms, 'rmse_avg',r_avg, 'rmse_global',r_glb, 'rmse_proposed',r_prp, ...
    'rmse_oracle',r_orc, 'rmse_kalman',r_kf, 'raw_err_time',e_prp, ...
    'raw_err_time_glb',e_glb, 'raw_err_time_kf',e_kf, 'raw_err_time_orc',e_orc);

if exist('signrank', 'file')
    [stats.wilcoxon_proposed_vs_kalman_p, ~, stats.wilcoxon_proposed_vs_kalman_stats] = signrank(r_prp, r_kf);
else
    stats.wilcoxon_proposed_vs_kalman_p = NaN; stats.wilcoxon_proposed_vs_kalman_stats = struct('signedrank', NaN);
end
stats.relative_improvement_vs_kalman_pct = (mean(r_kf) - mean(r_prp)) / mean(r_kf) * 100;
end
