% Run cluster-based permutation stats and plot beta dprime and beta
% criterion as a function of time (when network is ON vs. OFF)
% 
% JY (Aug, 2024)

%% User-defined keywords
% The K-state model of interest
if ~exist('K','var')
    K = input('Please specify the K-state model of interest (e.g., 8).\n', 's'); 
    K = str2double(K);
end

% The ith state of interest in this K-state model
if ~exist('ik','var')
    ik = input('Please specify the ith state of interest (e.g., 0).\n', 's'); 
    ik = str2double(ik);
end

%% Load the results
outfname = sprintf('GLMM_BS_pow_x_vpath_bestrun_%dth_of_%d_states.mat', ik, K);
load( outfname, 'stat','statBS','beta','tvec' );


%% Run permutation stats
cfg              = [];
cfg.tvec         = tvec;
cfg.tbeg         = -0.5;
cfg.tend         = 0.25;
cfg.clusteralpha = 0.05;
stat = glmm_timecourse_stat( cfg, stat, statBS );


%% Do the plot
% DEFINE the significance level
alpha = 0.05;

% DEFINE the colors
colors = struct('ON',[178,24,43]./255,...
                'OFF',[0.5,0.5,0.5]); %'OFF',[33,102,172]./255);

% PLOT both beta_dp and beta_c             
for iV = 1:2
    figure, hold on,
    %%%set(gcf,'units','centimeters','position',[0 0 6 4]);
    set(gcf,'units','centimeters','position',[0 0 9 4]);
    switch iV
        case 1, vname='Dp'; yname='\beta_d_'''; iON=1; iOFF=2; iInt=3;
        case 2, vname='C'; yname='\beta_c'; iON=4; iOFF=5; iInt=6;
    end

    %plot zero-line
    yline(0, 'k--');

    %plot the ON
    lineON = plot( tvec, beta.(vname)(:,1), '-',...
                    'Color',colors.ON,'linewidth',2.5);

    %plot the OFF
    lineOFF = plot( tvec, beta.(vname)(:,2), '-', ...
        'linewidth',1.5, 'Color',colors.OFF);
    
    %Set figure
    set(gca,'TickDir','out', 'FontSize',10);

    %Set yscale
    ymax = max( abs(beta.(vname)(:)) );
    yticks( [-0.1:0.05:0.1] );
    if (ymax*1.5)>0.1
        yticklabels( ["-0.1","     ", "0", "    ", "0.1"]);
    else %this prevents "-0.05" and "0.05" from changing the axis' position
        yticklabels( ["-0.1","-0.05", "0", "0.05", "0.1"]);
    end
    ylim( [-1,1].*ymax.*1.5 );
    YLim = ylim();
    ylabel( yname, 'FontSize',14 );

    %Set x-scale
    xlabel( 'Time (s)', 'FontSize',14);
    xlim( [tvec(1)-0.01, tvec(end)+0.01] );
    if tvec(end)<0.1
        xticks( -1:0.25:0 );
    else
        xticks( -1:0.25:1 );
    end
    
    % ================ Plot significance =============
    iSigLine = 0;
    
    %Plot significant ON
    if any(stat.Pvalcorrected(:,iON)<alpha)
        iSigLine = iSigLine + 1;
        ysig = YLim(1) + diff(YLim)*0.035*iSigLine;
        time_masked = tvec(stat.Pvalcorrected(:,iON)<alpha);
        midx = find( diff(time_masked)>(tvec(2)-tvec(1)) );
        midx = [midx, numel(time_masked), numel(time_masked)];
        for id = 1:(numel(midx)-1)
            if id==1
                curridx = 1:(midx(1));
            else
                curridx = (midx(id-1)+1):midx(id);
            end
            plot( time_masked(curridx), ones(size(curridx)).*ysig, '-', ...
                    'Color',colors.ON, 'linewidth',3);
    % % % %         plot( time_masked(curridx), var_masked(curridx), '-', ...
    % % % %             'Color',colors.ON, 'linewidth',4);
        end
    end
    
    %Plot significant OFF
    %{
    if any(stat.Pvalcorrected(:,iOFF)<alpha)
        iSigLine = iSigLine + 1;
        ysig = YLim(1) + diff(YLim)*0.035*iSigLine;
        time_masked = tvec(stat.Pvalcorrected(:,iOFF)<alpha);
        midx = find( diff(time_masked)>(tvec(2)-tvec(1)) );
        midx = [midx, numel(time_masked), numel(time_masked)];
        for id = 1:(numel(midx)-1)
            if id==1
                curridx = 1:(midx(1));
            else
                curridx = (midx(id-1)+1):midx(id);
            end
            plot( time_masked(curridx), ones(size(curridx)).*ysig, '-', ...
                    'Color',colors.OFF, 'linewidth',3);
        end
    end
    %}
    
    %Plot significant interaction between Power and Network ON-OFF
    if any(stat.Pvalcorrected(:,iInt)<alpha)
        iSigLine = iSigLine + 1;
        ysig = YLim(1) + diff(YLim)*0.035*iSigLine;
        t_masked = tvec( stat.Pvalcorrected(:,iInt)<alpha );
        midx = find( diff(t_masked)>(tvec(2)-tvec(1)) );
        midx = [midx, numel(t_masked), numel(t_masked)];
        for id = 1:(numel(midx)-1)
            if id==1
                curridx = 1:(midx(1));
            else
                curridx = (midx(id-1)+1):midx(id);
            end

            plot( t_masked(curridx), ones(size(curridx)).*ysig, 'k-', 'linewidth',3);
        end
    end
    
    %Legend
    %{
    legend([lineON,lineOFF],{'V=1','V=0'},...
        'NumColumns',2,'Location','northwest','FontSize',8);
    legend boxoff
    %}
    
    %Save the figure
    print(sprintf('GLMM_beta%s_bestrun_%dth_of_%d_states.png', vname, ik, K), '-dpng','-r400');
    
end


%% =============== sub-function ===============
function stat = cluster_based_permutation( stat, statBS, clusteralpha )
% Run cluster-based permutation test for time-series data.
%
% INPUT:
%   stat        : stats of the real data.
%   statBS      : stats of the permuted samples.
%   clusteralpha: cluster-forming alpha.
% 
% OUTPUT:
%   stat: stats of the real data, with field "Pvalcorrected"
%
% JY (Aug, 2024)

nPermute = size( statBS.Pval, 3 );

%{
% Define cluster alpha for the F-test
clusteralpha = 0.05;

% Define alpha cutoff for the identified clusters
alpha = 0.05;
%}

% Do the clustering and compute maxsum stats for the permuted
n_contrasts = size( statBS.Fstat, 2 );
maxsum_mat  = nan( n_contrasts, nPermute );
for cc = 1:n_contrasts
    for b = 1:nPermute
        
        [L, NUM] = bwlabel( statBS.Pval(:,cc,b)<clusteralpha );
        
        if any(L) > 0 %when there is at least one cluster
            sumF_vec = [];
            tmpF_vec = statBS.Fstat(:,cc,b);
            for ii = 1:NUM
                sumF_vec = [sumF_vec, sum( tmpF_vec(L==ii) ) ];
            end
            maxsum_mat(cc,b) = max(sumF_vec);
            
        else %when there is no cluster
            maxsum_mat(cc,b) = 0;
        end
    end
end

% Do the clustering and compute maxsum stats for the real data
STATS = cell( n_contrasts, 1 );
for cc = 1:n_contrasts
        
    [L, NUM] = bwlabel( stat.Pval(:,cc)<clusteralpha );
    
    if any(L) > 0 %when there is at least one cluster
        STATS{cc} = cell( NUM, 1 );
        tmpF_vec  = stat.Fstat(:,cc);
        for ii = 1:NUM
            STATS{cc}{ii}.mask         = (L==ii);
            STATS{cc}{ii}.clusterstats = sum( tmpF_vec(L==ii) );
            %%%STATS{cc}{ii}.pvalue       = 1 - mean( STATS{cc}{ii}.clusterstats > maxsum_mat(cc,:) );
            
            if STATS{cc}{ii}.clusterstats >= max(maxsum_mat(cc,:))
                STATS{cc}{ii}.pvalue = 0; %clusterstat for the real-data is the largest compared to those of permuted samples
            else
                [f,x] = ecdf( maxsum_mat(cc,:) );
                [x,I] = unique( x );
                p_interp = interp1( x, f(I), STATS{cc}{ii}.clusterstats ); %interpolation allows slightly better precision
                STATS{cc}{ii}.pvalue = 1 - p_interp;
            end
        end
    end    
end

% Make it easier to understand
stat.Pvalcorrected = ones( size(stat.Pval) );
for cc = 1:n_contrasts
    
    pval_vec = ones( size(stat.Pval,1), 1 );
    
    for ii = 1:numel( STATS{cc} )
        curr_stat = STATS{cc}{ii};
% % %         if curr_stat.pvalue < alpha
% % %             pval_vec( curr_stat.mask ) = curr_stat.pvalue;
% % %         end
        pval_vec( curr_stat.mask ) = curr_stat.pvalue;
    end
    stat.Pvalcorrected(:,cc) = pval_vec;
end

end

%% =============== sub-function ===============
function stat = glmm_timecourse_stat(cfg, stat, statBS)
% This is the high(er)-level wrapper function to perform cluster-based
% permutation test on the GLMM time courses.
% 
% INPUT:
%   cfg   : configuration, including:
%               .tvec (required): corresponding time points
%               .tbeg (optional): first time point for runing the stats
%               .tend (optional): last time point for runing the stats
%               .clusteralpha (optional): cluster-forming alpha.
%   stat  : stats of the real data.
%   statBS: stats of the permuted samples.
%  
% OUTPUT:
%   stat: output stats with field "Pvalcorrected" corresponding to each
%           time points in "cfg.tvec"
% 
% JY (Aug, 2024)

cfg.tbeg = ft_getopt(cfg, 'tbeg', cfg.tvec(1));
cfg.tend = ft_getopt(cfg, 'tend', cfg.tvec(end));

cfg.clusteralpha = ft_getopt(cfg, 'clusteralpha', 0.05);

idx = find( cfg.tvec>=cfg.tbeg & cfg.tvec<=cfg.tend );
stmp = structfun( @(x) x(idx,:), stat, 'uniformoutput', false );
btmp = structfun( @(x) x(idx,:,:), statBS, 'uniformoutput', false );
sout = cluster_based_permutation( stmp, btmp, cfg.clusteralpha );

stat.Pvalcorrected = ones( size(stat.Pval) );
stat.Pvalcorrected(idx,:) = sout.Pvalcorrected;

end
