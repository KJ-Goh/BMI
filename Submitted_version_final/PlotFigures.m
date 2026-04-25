od='./experiment_outputs/';

load(fullfile(od,'main_results_bundle.mat'));
load(fullfile(od,'ablation_bundle.mat'));
tCls=table();
if exist(fullfile(od,'Table_targeted_classifier_comparison.csv'),'file')
    tCls=readtable(fullfile(od,'Table_targeted_classifier_comparison.csv'),'TextType','string');
end
if exist(fullfile(od,'patch_bundle.mat'),'file')
    load(fullfile(od,'patch_bundle.mat')); fprintf('Bundles and patched data loaded.\n');
else
    fprintf('Bundles loaded (no patch found).\n');
end

set(0,'DefaultAxesFontName','Times New Roman','DefaultAxesFontSize',13,...
    'DefaultTextFontName','Times New Roman','DefaultTextFontSize',13,...
    'DefaultLineLineWidth',1.5,'DefaultLineMarkerSize',7,...
    'DefaultAxesLineWidth',1.0,'DefaultAxesBox','on');

c.prp=[0.20 0.50 0.80]; c.orc=[0.20 0.70 0.30]; c.glb=[0.84 0.15 0.16];
c.kf=[0.93 0.69 0.13]; c.avg=[0.50 0.50 0.50]; c.acc=[0.84 0.15 0.16];
c.aux=[0.00 0.45 0.20];

%% Fig 1
disp('Plotting Fig 1 (Overall Performance)...');
f1=fig([7.50 4.90]); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

ax1=nexttile; hold on; grid on; tt=stats.times(:)';
cs={...
    {'Oracle',stats.err_time_orc_mean(:)',stats.err_time_orc_std(:)',c.orc,'--',1.8,'^',2:3:numel(tt),'w',0.08},...
    {'Global PCR',stats.err_time_glb_mean(:)',stats.err_time_glb_std(:)',c.glb,'-',2.0,'none',[],c.glb,0.10},...
    {'Global Kalman Filter',stats.err_time_kf_mean(:)',stats.err_time_kf_std(:)',c.kf,'-',2.2,'none',[],c.kf,0.10},...
    {'Proposed',stats.err_time_mean(:)',stats.err_time_std(:)',c.prp,'-',2.9,'o',1:2:numel(tt),c.prp,0.12}};
hs=gobjects(0); labs={};
for i=1:numel(cs)
    x=cs{i};
    fill([tt fliplr(tt)],[x{2}+x{3} fliplr(x{2}-x{3})],x{4},'FaceAlpha',x{10},'EdgeColor','none','HandleVisibility','off');
    hs(end+1)=plot(tt,x{2},'LineStyle',x{5},'Color',x{4},'LineWidth',x{6},'Marker',x{7},...
        'MarkerFaceColor',x{9},'MarkerEdgeColor',x{4},'MarkerIndices',x{8});
    labs{end+1}=x{1};
end
xlabel('Time (ms)','Interpreter','latex','FontSize',16); ylabel('Mean Position Error (cm)','Interpreter','latex','FontSize',16);
title('(a) Decoding RMSE over Time','Interpreter','latex','FontSize',18); xlim([min(tt) max(tt)]); pbaspect([4 3 1]); set(gca,'FontSize',15);

nexttile; hold on; grid on;
rp=stats.rmse_proposed(:); ro=stats.rmse_oracle(:); rg=stats.rmse_global(:); ra=stats.rmse_avg(:); rk=stats.rmse_kalman(:);
pg=signrank(rp,rg); pk=signrank(rp,rk); pa=signrank(rp,ra);
bd={ra,rg,rk,rp,ro}; bn={'Mean Traj','Global PCR','Global KF','Proposed','Oracle'}; av=vertcat(bd{:});
for i=1:numel(bd)
    col=gc(bn{i},c);
    boxchart(i*ones(size(bd{i}(:))),bd{i}(:),'BoxFaceColor',col,'BoxEdgeColor',col,'MarkerStyle','none','LineWidth',1.1);
end
xticks(1:numel(bn)); xticklabels(bn); ylabel('RMSE (cm)','Interpreter','latex','FontSize',16); title('(b) Overall RMSE Distribution','Interpreter','latex','FontSize',18); set(gca,'FontSize',15);

q1=zeros(1,numel(bd)); q3=q1; mu=q1;
for i=1:numel(bd), q1(i)=quantile(bd{i},0.25); q3(i)=quantile(bd{i},0.75); mu(i)=mean(bd{i}); end
tw=max(q3+1.5.*(q3-q1)); yt=tw*1.05; ys=tw*0.08; ylim([min(av)*0.95, yt+3.8*ys]);
for i=1:numel(bd)
    text(i,q3(i)+0.55*ys,sprintf('%.2f',mu(i)),'HorizontalAlignment','center','FontSize',13,'Interpreter','latex','Color',gc(bn{i},c));
end
ip=find(strcmp(bn,'Proposed')); ik=find(strcmp(bn,'Global KF')); ig=find(strcmp(bn,'Global PCR')); ia=find(strcmp(bn,'Mean Traj'));
sig(ik,ip,yt,ys,pk); sig(ig,ip,yt+1.2*ys,ys,pg); sig(ia,ip,yt+2.4*ys,ys,pa);
xtickangle(20); pbaspect([4 3 1]);

lh=gobjects(0); ln={};
for i=1:numel(bn)
    fn=en(bn{i}); if any(strcmp(ln,fn)), continue; end
    j=find(strcmp(labs,fn),1);
    lh(end+1)=iff(isempty(j),plot(ax1,nan,nan,'-','Color',gc(bn{i},c),'LineWidth',1.8),hs(j));
    ln{end+1}=fn;
end
lgd=legend(ax1,lh,ln,'Orientation','horizontal','FontSize',12,'Interpreter','latex');
lgd.Layout.Tile='north'; lgd.NumColumns=min(4,numel(ln));
sv(f1,od,'Fig1_Performance'); fprintf('Fig1 saved.\n');

%% Fig 2
disp('Plotting Fig 2 (Subspace Dimensions)...');
f2=fig([7.16 5.5]); tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile([1 2]); hold on; grid on;
fr=r_grid(~isinf(r_grid)); fx=r_plot(end);
eb(r_plot,r_prp_mean,r_prp_std,'-o',c.prp);
eb(r_plot,r_orc_mean,r_orc_std,'-^',c.orc);
eb(r_plot,r_glb_mean,r_glb_std,'-s',c.glb);
yline(stats.rmse_avg_mean,'--','Color',c.avg,'LineWidth',1.2); xline(fx,':k','LineWidth',1);
xticks([fr,fx]); xticklabels([arrayfun(@num2str,fr,'UniformOutput',false),{'Full'}]);
xlabel('PCR Rank $r$','Interpreter','latex'); ylabel('RMSE (cm)','Interpreter','latex');
title('(a) Regression Subspace Rank','Interpreter','latex');
legend({'Proposed','Oracle','Global PCR','Mean Traj'},'Location','best','FontSize',10,'Interpreter','latex');
pbaspect([4 1.5 1]);

for k=1:2
    nexttile; hold on; grid on;
    yyaxis left; set(gca,'YColor','k');
    if k==1
        eb(Mpca_list,sw_mpca_rmse_mean,sw_mpca_rmse_std,'-o',c.prp);
        yyaxis right; set(gca,'YColor',c.acc);
        eb(Mpca_list,sw_mpca_acc_mean,sw_mpca_acc_std,'-s',c.acc);
        xlabel('PCA Dimension $M_{PCA}$','Interpreter','latex'); title('(b) PCA Subspace Dimension','Interpreter','latex');
    else
        eb(Mlda_list,sw_mlda_rmse_mean,sw_mlda_rmse_std,'-o',c.prp);
        yyaxis right; set(gca,'YColor',c.acc);
        eb(Mlda_list,sw_mlda_acc_mean,sw_mlda_acc_std,'-s',c.acc);
        xlabel('LDA Dimension $M_{LDA}$','Interpreter','latex'); title('(c) LDA Feature Dimension','Interpreter','latex');
    end
    ylabel('Accuracy','Color',c.acc,'Interpreter','latex'); ylim([0 1.05]);
    yyaxis left; ylabel('RMSE (cm)','Interpreter','latex'); pbaspect([4 3 1]);
end
sv(f2,od,'Fig2_Subspace_Dimensions'); fprintf('Fig2 saved.\n');

%% Fig 3
disp('Plotting Fig 3 (Temporal Resolution)...');
f3=fig([7.60 4.25]); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nr=length(dt_reg_grid); [~,br]=min(a1r_mean);
nexttile;
cc=repmat([0.75 0.75 0.75],nr,1); cc(br,:)=c.prp;
yyaxis left; set(gca,'YColor','k');
hb=bar(1:nr,a1r_mean,0.5,'FaceColor','flat','EdgeColor','k','CData',cc); hold on;
he=errorbar(1:nr,a1r_mean,a1r_std,'k.','CapSize',5); set(he,'HandleVisibility','off');
ylabel('RMSE (cm)','Interpreter','latex','FontSize',16);
yyaxis right; set(gca,'YColor',c.aux);
hl=plot(1:nr,a1r_spk,'-^','Color',c.aux,'MarkerFaceColor',c.aux);
xticks(1:nr); xticklabels(arrayfun(@(x)sprintf('%d',x),dt_reg_grid,'UniformOutput',false));
xlabel('Regression Window $\Delta t_{reg}$ (ms)','Interpreter','latex','FontSize',16);
ylabel('Mean Spikes / Bin','Color',c.aux,'Interpreter','latex','FontSize',16);
grid on; title('(a) Regression Window','Interpreter','latex','FontSize',18); pbaspect([4 3 1]); set(gca,'FontSize',15);

nexttile; hold on; grid on;
nc=length(dtclass_list); [~,bc]=min(sw_dtc_rmse_mean);
cc2=repmat([0.75 0.75 0.75],nc,1); cc2(bc,:)=c.prp;
yyaxis left; set(gca,'YColor','k');
bar(1:nc,sw_dtc_rmse_mean,0.5,'FaceColor','flat','EdgeColor','k','CData',cc2);
he=errorbar(1:nc,sw_dtc_rmse_mean,sw_dtc_rmse_std,'k.','CapSize',5); set(he,'HandleVisibility','off');
ylabel('RMSE (cm)','Interpreter','latex','FontSize',16);
sp=iff(exist('a1c_spk','var'),a1c_spk,dtclass_list*(a1r_spk(2)/dt_reg_grid(2)));
yyaxis right; set(gca,'YColor',c.aux);
plot(1:nc,sp,'-^','Color',c.aux,'MarkerFaceColor',c.aux);
xticks(1:nc); xticklabels(arrayfun(@(x)sprintf('%d',x),dtclass_list,'UniformOutput',false));
xlabel('Classification Window $\Delta t_{class}$ (ms)','Interpreter','latex','FontSize',16);
xlim([0.45 nc+0.5]);
ylabel('Mean Spikes / Bin','Color',c.aux,'Interpreter','latex','FontSize',16);
title('(b) Classification Window','Interpreter','latex','FontSize',18); pbaspect([4 3 1]); set(gca,'FontSize',15);

lgd=legend([hb,hl],{'Performance Metric (RMSE)','Mean Spikes / Bin'},'Orientation','horizontal','Interpreter','latex','FontSize',12);
lgd.Layout.Tile='north';
sv(f3,od,'Fig3_Temporal_Resolution'); fprintf('Fig3 saved.\n');

%% Fig 4
if exist('hm_rmse_reg','var')&&exist('hm_acc_cls','var')
    disp('Plotting Fig 4 (Heatmaps - Regression and Classification)...');
    f4=fig([7.16 4.0]); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile; imagesc(hm_rmse_reg); cb=colorbar; cb.Label.Interpreter='latex'; colormap(gca,parula);
    xticks(1:length(hm_dt_reg)); xticklabels(arrayfun(@(x)sprintf('%d',x),hm_dt_reg,'UniformOutput',false));
    yt=arrayfun(@num2str,hm_r,'UniformOutput',false); yt{end}='Full';
    yticks(1:length(hm_r)); yticklabels(yt);
    xlabel('Regression Window $\Delta t_{reg}$ (ms)','Interpreter','latex');
    ylabel('PCR Rank $r$','Interpreter','latex'); title('(a) Regression RMSE Coupling','Interpreter','latex');
    for i=1:length(hm_r), for j=1:length(hm_dt_reg), v=hm_rmse_reg(i,j); text(j,i,sprintf('%.1f',v),'HorizontalAlignment','center','FontSize',10,'Color',iff(v>15,'k','w'),'FontWeight','bold'); end, end
    pbaspect([1 1 1]);

    nexttile; imagesc(hm_acc_cls); cb=colorbar; cb.Label.Interpreter='latex'; colormap(gca,parula);
    xticks(1:length(hm_dt_cls)); xticklabels(arrayfun(@(x)sprintf('%d',x),hm_dt_cls,'UniformOutput',false));
    yticks(1:length(hm_M)); yticklabels(arrayfun(@num2str,hm_M,'UniformOutput',false));
    xlabel('Classification Window $\Delta t_{class}$ (ms)','Interpreter','latex');
    ylabel('PCA Dimension $M_{PCA}$','Interpreter','latex'); title('(b) Classification Acc Coupling','Interpreter','latex');
    for i=1:length(hm_M), for j=1:length(hm_dt_cls), v=hm_acc_cls(i,j); text(j,i,sprintf('%.3f',v),'HorizontalAlignment','center','FontSize',10,'Color',iff(v<0.6,'w','k'),'FontWeight','bold'); end, end
    pbaspect([1 1 1]);

    sv(f4,od,'Fig4_Heatmap'); fprintf('Fig4 saved.\n');
end

%% Fig 5
disp('Plotting Fig 5 (Prior Ablation)...');
f5=fig([9.20 4.65]); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
if ~exist('cond_labels','var'), cond_labels={'Chance','PCA','PCA-LDA','PCA-KNN','PCA-SVM','Oracle'}; end
if ~isempty(tCls)
    cond_labels=cellstr(tCls.Model)'; 
    am=tCls.Mean_Accuracy; as=tCls.Std_Accuracy;
    rm=tCls.Mean_RMSE_cm; rs=tCls.Std_RMSE_cm;
    cc=cell2mat(cellfun(@(x)ccf(x,c),cond_labels','uni',0));
else
    am=mean(acc_cond,2); as=std(acc_cond,0,2);
    rm=mean(rmse_cond,2); rs=std(rmse_cond,0,2);
    if size(acc_cond,1)==4
        cond_labels={'Chance','PCA','PCA-LDA','Oracle'};
        cc=[c.avg;0.9 0.6 0.2;c.prp;c.orc];
    else
        cc=lines(size(acc_cond,1));
    end
end
dn=sn(cond_labels); n=numel(cond_labels); x=1:n;
Y={am,rm}; S={as,rs};
YL={'Accuracy','RMSE (cm)'};
TT={'(a) Classifier Accuracy Comparison','(b) Decoder RMSE by Classifier'};
YM={[0 1.18],[0 max(rm+rs)*1.18]};
for k=1:2
    nexttile; hold on; grid on;
    y=Y{k}; s=S{k}; off=iff(k==1,0.02,0.02*max(rm));
    bar(x,y,0.5,'FaceColor','flat','CData',cc,'EdgeColor','k');
    errorbar(x,y,s,'k.','CapSize',5);
    text(x,y+s+off,compose('%.2f',y),'HorizontalAlignment','center','FontSize',13,'Interpreter','latex');
    xticks(x); xticklabels(dn); xtickangle(16);
    xlim([0.35 n+1.45]); ylim(YM{k});
    ylabel(YL{k},'Interpreter','latex','FontSize',16);
    title(TT{k},'Interpreter','latex','FontSize',18);
    set(gca,'PositionConstraint','outerposition')
end
sv(f5,od,'Fig5_Prior_Ablation'); fprintf('Fig5 saved.\n');
%% Fig 6
disp('Plotting Fig 6 (Data Efficiency)...');
f6=figure('Visible','off','Units','inches','Position',[1 1 3.5 3.2],'PaperUnits','inches','PaperSize',[3.5 3.2]); hold on; grid on;
eb(sP_list,a2_avg_mean,a2_avg_std,'--d',c.avg,1.2);
eb(sP_list,a2_glb_mean,a2_glb_std,'--s',c.glb,1.2);
eb(sP_list,a2_prp_mean,a2_prp_std,'-o',c.prp);
eb(sP_list,a2_orc_mean,a2_orc_std,'-^',c.orc);
xlabel('Training Fraction $s_P$','Interpreter','latex'); ylabel('RMSE (cm)','Interpreter','latex');
title('Training Data Efficiency','Interpreter','latex');
legend({'Mean Traj','Global PCR','Proposed','Oracle'},'Location','best','FontSize',9,'Interpreter','latex');
pbaspect([4 3 1]); sv(f6,od,'Fig6_Data_Efficiency'); fprintf('Fig6 saved.\n');

%% Fig 7
if exist('rob_prp_mean','var')
    disp('Plotting Fig 7 (Robustness)...');
    f7=fig([7.16 3.5]); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile; hold on; grid on;
    eb(1:length(snr_db_list),rob_prp_mean,rob_prp_std,'-o',c.prp);
    eb(1:length(snr_db_list),rob_glb_mean,rob_glb_std,'-s',c.glb);
    xticks(1:length(snr_db_list)); xticklabels({'5 dB','10 dB','20 dB','30 dB','Clean'});
    xlabel('Input SNR','Interpreter','latex'); ylabel('RMSE (cm)','Interpreter','latex');
    title('(a) Degradation: Gaussian Noise','Interpreter','latex');
    legend({'Proposed','Global PCR'},'Location','northeast','FontSize',9,'Interpreter','latex'); pbaspect([4 3 1]);

    nexttile; hold on; grid on;
    eb(miss_list*100,miss_prp_mean,miss_prp_std,'-o',c.prp);
    eb(miss_list*100,miss_glb_mean,miss_glb_std,'-s',c.glb);
    xlabel('Missing Rate (\%)','Interpreter','latex'); ylabel('RMSE (cm)','Interpreter','latex');
    title('(b) Degradation: Missing Spikes','Interpreter','latex');
    legend({'Proposed','Global PCR'},'Location','northwest','FontSize',9,'Interpreter','latex'); pbaspect([4 3 1]);

    sv(f7,od,'Fig7_Robustness'); fprintf('Fig7 saved.\n');
end

function f=fig(sz)
f=figure('Visible','off','Units','inches','Position',[1 1 sz],'PaperUnits','inches','PaperSize',sz);
end

function sv(f,od,n)
print(f,fullfile(od,[n '.png']),'-dpng','-r300');
print(f,fullfile(od,[n '.pdf']),'-dpdf','-r300');
end

function sig(a,b,y,s,p)
plot([a b],[y y],'k-'); plot([a a],[y-0.2*s y],'k-'); plot([b b],[y-0.2*s y],'k-');
    text(mean([a b]),y+0.4*s,['$p=' sprintf('%.3f',p) '$' gs(p)],'HorizontalAlignment','center','FontSize',13,'Interpreter','latex');
end

function x=ccf(n,c)
s=lower(strtrim(char(n)));
if contains(s,'chance')
    x=c.avg;
elseif contains(s,'pca-lda')
    x=c.prp;
elseif contains(s,'pca-knn')
    x=[0.55 0.35 0.75];
elseif contains(s,'pca-svm')
    x=[0.15 0.65 0.65];
elseif contains(s,'oracle')
    x=c.orc;
elseif contains(s,'pca')
    x=[0.9 0.6 0.2];
else
    x=[0.65 0.65 0.65];
end
end

function x=gc(n,c)
s=lower(strtrim(char(n)));
if contains(s,'mean')
    x=c.avg;
elseif contains(s,'global pcr')
    x=c.glb;
elseif contains(s,'global kf')||contains(s,'kalman')
    x=c.kf;
elseif contains(s,'oracle')
    x=c.orc;
elseif contains(s,'proposed')
    x=c.prp;
else
    x=[0.65 0.65 0.65];
end
end

function x=sn(a)
x=cell(size(a));
for i=1:numel(a)
    if strcmp(string(a{i}),"Oracle Lower Bound")||strcmp(string(a{i}),"Oracle LB"), x{i}='Oracle';
    else, x{i}=char(a{i});
    end
end
end

function x=en(a)
if strcmp(string(a),"Global KF")
    x='Global Kalman Filter';
elseif strcmp(string(a),"Oracle LB")||strcmp(string(a),"Oracle Lower Bound")
    x='Oracle';
else
    x=char(a);
end
end

function x=gs(p)
x=''; if p<0.05, x=[x '*']; end; if p<0.01, x=[x '*']; end; if p<0.001, x=[x '*']; end
end

function y=iff(c,a,b)
if c, y=a; else, y=b; end
end

function eb(x,y,e,m,col,lw)
if nargin<6, lw=1.5; end
errorbar(x,y,e,m,'Color',col,'MarkerFaceColor',col,'CapSize',3,'LineWidth',lw);
end
