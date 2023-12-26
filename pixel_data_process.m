usr_dir = "D:\research_new";
addpath(genpath(usr_dir + "\MFM_DPFA_clean"));
addpath(genpath(usr_dir + "\data"));

%% just see the data tonight...
load('719161530_spikes.mat') % spike times for each unit
tab = readtable('719161530_units.csv'); % meta-data
subset = find(tab.snr>3 & cellfun(@length,Tlist')>1000);

% tt = tab.ecephys_structure_acronym(subset);
% y=unique(tt);
% for k=1:numel(y)
%   freq(k)=sum(strcmp(tt,y(k)));
% end
% [freq_sorted,idx] = sort(freq, 'descend');
% [y(idx) num2cell(freq_sorted')]

region_sel = ["CA1" "VISp" "LP"]; % ["CA1" "LGd" "VISp" "LP"];
idx_region = find(ismember(string(tab.ecephys_structure_acronym),region_sel));
subset2 = intersect(subset, idx_region);
N = length(subset2);
lab_all_str = string(tab.ecephys_structure_acronym);
[lab_num_sub2, clusIdx] = findgroups(lab_all_str(subset2));

dt = 0.5;
Tstart = 2822.161967;
Tend = 3182.462857;
Tmark = [2852.187037 3152.437777];
T = floor((Tend-Tstart)/dt);
Tmark_bin = floor((Tmark - Tstart)/dt);

% trial 1:
% 2190.634543	2221.660447	31.02590428	spontaneous	null
% 2221.660447	2822.161967	600.50152	natural_movie_three	3
% 2822.161967	2852.187037	30.02507	spontaneous	null
% 2852.187037	3152.437777	300.25074	natural_movie_one	4

% trial 2:
% 10	2822.161967	2852.187037	30.02507	spontaneous	null
% 11	2852.187037	3152.437777	300.25074	natural_movie_one	4
% 12	3152.437777	3182.462857	30.02508	spontaneous	null

% trial 3:
% 26	7680.268867	7710.293937	30.02507	spontaneous	null
% 27	7710.293937	8010.544677	300.25074	natural_movie_one	12
% 28	8010.544677	8040.569727	30.02505	spontaneous	null

% trial 4:
% 14	3781.963503	4083.215117	301.2516143	spontaneous	null
% 15	4083.215117	4683.716567	600.50145	natural_movie_three	6



Yraw = zeros(N, T);
for n = 1:N
    for k = 1:T
        Yraw(n,k) = sum((Tlist{subset2(n)} > (dt*(k-1)) + Tstart) &...
            (Tlist{subset2(n)} < (dt*k + Tstart)));
    end
end

[Lab, idx] = sort(lab_num_sub2);
Y = Yraw(idx,:);

idx_use = sum(Y,2)>=T*dt;
Y2 = Y(idx_use,:);
Lab2 = Lab(idx_use,:);
N2 = size(Y2,1);


figure(1)
clusterPlot(Y2, Lab2')
for ll = 1:size(Tmark_bin,2)
    xline(Tmark_bin(ll), 'k--', 'LineWidth', 2);
end
title(clusIdx)

pixel_spk = figure;
hold on
imagesc(Y2)
for ll = 1:size(Tmark_bin,2)
    xline(Tmark_bin(ll), 'r--', 'LineWidth', 2);
end
n_tmp = 0;
ytickPos = zeros(1, length(unique(Lab2)));
for mm = sort(unique(Lab2))'
   n_pre = n_tmp;
   ytickPos(mm) = n_tmp + 0.5*sum(Lab2 == mm);
   
   n_tmp_ = n_tmp + 0.5*sum(Lab2 == mm);
   n_tmp = n_tmp + sum(Lab2 == mm);
   yline(n_tmp+0.5, 'r--', 'LineWidth', 2);
   
end
yticks(ytickPos)
yticklabels(clusIdx)
xlim([0 T])
ylim([0 N2])
colormap(flipud(gray))
colorbar()
hold off
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')

set(pixel_spk,'PaperUnits','inches','PaperPosition',[0 0 2.75 2])
% saveas(pixel_spk, "pixel_spk.svg")
% saveas(pixel_spk, "pixel_spk.png")


Lab_t = zeros(T,1);
Tmark_bin_2 = [0 Tmark_bin T];
states = [0 1 0];
start_tmp = 1;
for kk = 1:(size(Tmark_bin_2,2)-1)
    Lab_t((Tmark_bin_2(kk) + 1):Tmark_bin_2(kk+1)) = states(kk);
end

% writematrix(Y2, "Y.csv")
% writematrix(Lab2, "Lab_neuron.csv")
% writematrix(Lab_t, "Lab_states.csv")

%%
mean(Lab_t==0)
% mean(Lab_t==1)





