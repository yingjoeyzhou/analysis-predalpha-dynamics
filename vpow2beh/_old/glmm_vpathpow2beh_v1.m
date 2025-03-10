%
% Run GLMM to see if pre-stim alpha power predicts d' or criterion.
%
% In the GLMM, vpath was used for trial selection and as a predicter.
%
% JY (Jan 2024)

% clearvars; close all; clc;


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

% The threshold used 
if ~exist('vpath_thresh','var')
    vpath_thresh = input('Please specify the cutoff for vpath (value between 0 and 1).\n', 's');
    vpath_thresh = str2double(vpath_thresh);
end


% Display in the command window what we do
fprintf('We will apply GLMM (with vpath_thresh=%s) of state #%d of the %d-states model.\n', ...
                    num2str(vpath_thresh), ik, K);


%% filename and directories
% directory and fnames
CODE_DIR = '/Volumes/ExtDisk/analysis_DondersData/3018041.02/vpow2beh';
CSV_DIR  = fullfile( CODE_DIR, 'csv_stateprob_parc9975' );
addpath( genpath(CODE_DIR) );

% csv filename 
csv_fname = sprintf('group_pow_and_vpath_%dstatesmodel_state%d.csv', K, ik);
disp( csv_fname );

%% read in the csv file and take care of the renamed column names
%read in the csv file as a table
data = readtable( csv_fname );

%the time points where we estimated alphapow from
[tvec, powtnames] = fetch_tvec_from_table( data );


%% Fit the GLMM
vec_primetype= cellfun(@(x) x(1)=='c', data.condition, 'uniformoutput',true); %gives 1 for conservertive, 0 for liberal
vec_yes      = data.report;
vec_presence = data.stim_presence;
vec_subject  = data.subject;
n = numel( unique(vec_subject) );
X = horzcat( vec_yes, vec_presence, vec_subject, vec_primetype );

stat = struct('Dp',[], 'Crit',[]);

primetype = {'cons','lib'};

for iT = 1:numel(tvec)
    
    pname = powtnames{iT};
    vname = strrep( pname, 'pow','vpath');
    alpha = data.(pname);
    vpath = data.(vname);
    
    %display the total number of trials
    %%%%%%%%%%%%===========JY edits: June 24, 2024
    %%%%%%%%%%%%sel_vpath = (vpath > vpath_thresh);
    sel_vpath = vpath==0;
    
    %show progress
    fprintf('\n\nWorking on time %s with %d trials...\n',...
        num2str( round(tvec(iT),3) ), sum(sel_vpath) );
    
    %index to exclude
    idx_to_exclude = ~sel_vpath;
    
    %normalize within subject
    for iSub = 1:n
        sel = (vec_subject==iSub);
        alpha(sel) = zscore(log(alpha(sel)));
    end
    
    %loop through the two prime types
    coef_table = cell(2,1);
    cov_table  = cell(2,1);
    
    t = array2table( horzcat(X,alpha), ...
                        'VariableNames',{'y', 'presence', 'subject', 'primetype', 'alphapow'});
    
    % Test the effect of alphapow on y (i.e., probability report yes) while
    % controling for stimulus presence and test for interaction effects.
    % Model random subject intercepts and slopes.
    % Intercept is modeled implicitly both at the group and subject level.
    t.presence  = categorical( t.presence ); %dummy-code
    t.subject   = categorical( t.subject ); %dummy-code
    t.primetype = categorical( t.primetype ); %dummy-code
    m = fitglme( t, 'y ~ presence * alphapow + (presence*alphapow | subject)',...
            'Distribution', 'Binomial',...
            'Link', 'probit',...
            'Exclude', idx_to_exclude);
    
    % The beta coefficients
    [dp_beta, dp_se, c_beta, c_se] = compute_betas(m);

    % F-stats of these beta coefficients 
    [PVAL,F,DF1,DF2] = coefTest( m, [0 0 0 1] );
    fprintf( '\ndprime effect: pval = %s', num2str(PVAL) );
    stat.Dp = vertcat( stat.Dp, [dp_beta, dp_se, PVAL,F,DF1,DF2] );

    [PVAL,F,DF1,DF2] = coefTest( m, [0 0 1 1/2] );
    fprintf( '\ncriterion effect: pval = %s', num2str(PVAL) );
    stat.Crit = vertcat( stat.Crit, [c_beta, c_se, PVAL,F,DF1,DF2] );
    
end



%% Save the results
save(sprintf('GLMM_pow_and_vpath_bestrun_%dth_of_%d_states.mat', ik, K));



%% Do the plot
close all;
cmaps = struct('cons',[123,50,148]./255,...
               'lib', [0,136,55]./255);
addpath('/Users/joeyzhou/Documents/MATLAB/shadedErrorBar');
for iV = 1:2
    figure, hold on,
    set(gcf,'units','centimeters','position',[0 0 6 4]);
    switch iV
        case 1, vname='Dp'; yname='Beta_d_''';
        case 2, vname='Crit'; yname='Beta_c';
    end
    
    %{
    maxabs = max( abs([stat.cons.(vname)(:,1)-stat.cons.(vname)(:,2); ...
                        stat.cons.(vname)(:,1)+stat.cons.(vname)(:,2); ...
                        stat.lib.(vname)(:,1)-stat.lib.(vname)(:,2); ...
                        stat.lib.(vname)(:,1)+stat.lib.(vname)(:,2)]) );
    yrange = maxabs*2;
    %}
    
    %JY: hard-coded
    maxabs = 0.3;
    yrange = maxabs * 2;
    
    % ========= start plotting ============
    %show zero line
    plot( tvec, zeros(size(tvec)), 'k--');

    %plot estimated beta
    plot( tvec, stat.(vname)(:,1), 'linewidth',1, 'color', 'r' );

    %plot error around beta
    shadedErrorBar( tvec, stat.(vname)(:,1), stat.(vname)(:,2),...
            'lineProps',{'color','r'}, 'transparent',true );

    %show significance
    mask = stat.(vname)(:,3)<0.05;
    if any(mask)
        ypos = -1.1*maxabs+0.02*yrange;
        plot( tvec(mask), ypos*ones(sum(mask),1), '.', 'markerfacecolor', 'k',...
                'markersize',8, 'markeredgecolor','k');
    end

    %adjust the y-scale
    ylim( [-1.1, 1.1].*maxabs );
    ylabel( yname, 'FontSize',10 );

    %adjust the x-scale
    xlabel( 'Time (s)', 'FontSize',10);
    if tvec(end)<0.1
        xticks( -1:0.25:0 );
    else
        xticks( -1:0.5:1 );
    end
    xlim( [tvec(1)-0.05, tvec(end)+0.05] );

    %adjust the plot
    set(gca,'TickDir','out', 'FontSize',8);

    % Save the figure
    print(sprintf('GLMM_beta%s_bestrun_%dth_of_%d_states.png', vname, ik, K), '-dpng','-r400');
    
end





%% ============ sub-function =============
function [tvec, tnames] = fetch_tvec_from_table( data )

mask = startsWith( data.Properties.VariableNames, 'powt');

tnames = data.Properties.VariableNames(mask);

tvec = [];
for ii = 1:numel(tnames)
    tname = tnames{ii};
    
    % ======= JY:hard-coding =======
    idx = strfind( tname, '_' );
    str = '';
    
    %check if there is a minus sign: hardcoded "__"
    if any( strfind( tname, '__') ); str=[str,'-']; end
    
    %first digit
    str = [str, tname(idx(end-1)+1)];
    
    %add the "."
    str = [str, '.'];
    
    %second digit
    str = [str, tname((idx(end)+1):end)];
    
    tnum = str2double( str ); %disp( tnum );
    tvec = [tvec, tnum];
    % ======= JY:hard-coding =======
end


end

%% ============= sub-function ============
function [dp_beta, dp_se, c_beta, c_se] = compute_betas(m)

coef = table2cell( dataset2table( m.Coefficients ) );
covs = m.CoefficientCovariance;

colEst = 2;
colUB  = 8;
colLB  = 7;
colDF  = 5;
colSE  = 3;

dp_beta = coef{4, colEst}; %JY: hard-coded
dp_lb   = coef{4, colLB}; %JY: hard-coded
dp_ub   = coef{4, colUB}; %JY: hard-coded
dp_se   = coef{4, colSE}; %JY: hard-coded

N      = coef{3, colDF};
c_beta = -1*( coef{3,colEst} + 1/2*coef{4,colEst} ); %JY: hard-coded
c_se   = sqrt( 1*covs(3,3) + (1/2).^2 * covs(4,4) + 2*1*1/2*covs(3,4) );

% dp_ci  = dp_se * 1.96;
% c_ci   = c_se * 1.96;
end

%% ============ sub-function: NOT USED ============
function betas = compute_and_plot_betas( coef_table, cov_table, plotme )

coef = coef_table;
covs = cov_table;

if nargin==2, plotme=false; end

% % cc = cbrewer('div', 'PRGn', 4, 'pchip');
% % cc = vertcat(cc(1,:), cc(end,:));
cc = [ [123,50,148]./255;...
        [0,136,55]./255 ];
betas = [];
betas.Dp = cell(2,1);
betas.Crit = cell(2,1);
    
for iPrm = 1:2
    switch iPrm
        case 1, offset = -0.05;
        case 2, offset = 0.05;
    end

    %====== betas and the CI =======
    colEst = 2;
    colUB  = 8;
    colLB  = 7;
    colDF  = 5;
    colSE  = 3;
    
    dp_beta = coef{iPrm}{4, colEst}; %JY: hard-coded
    dp_lb   = coef{iPrm}{4, colLB}; %JY: hard-coded
    dp_ub   = coef{iPrm}{4, colUB}; %JY: hard-coded
    dp_se   = coef{iPrm}{4, colSE}; %JY: hard-coded
    
    N      = coef{iPrm}{3, colDF};
    c_beta = -1*( coef{iPrm}{3,colEst} + 1/2*coef{iPrm}{4,colEst} ); %JY: hard-coded
    c_se   = sqrt( 1*covs{iPrm}(3,3) + (1/2).^2 * covs{iPrm}(4,4) + 2*1*1/2*covs{iPrm}(3,4) );
    
    dp_ci  = dp_se * 1.96;
    c_ci   = c_se * 1.96;
    
    
    %====== plot ========
    if plotme
        figure(1000); hold on,
        set(gcf,'units','centimeters','position',[0 0 8 6]./2);
        if iPrm==1
            plot( [0 0], [0.5, 2.5], 'k:' );
            set(gca,'TickDir','out','XColor','k','YColor','k','fontsize',6,'fontname','arial','linewidth',1);
        end
        ddd = errorbar( dp_beta, 1+offset, [], [], dp_ci, dp_ci, '.','markersize',20,'capsize',0,'color',cc(iPrm,:));
        ccc = errorbar( c_beta, 1.5+offset, [], [], c_ci, c_ci, '.','markersize',20,'capsize',0,'color',cc(iPrm,:));
        ylim( [0.8, 1.8] );
        yticks( [1,1.5] );
        yticklabels( {'\beta_d_''', '\beta_c'} );
        xlim( [-2.2,2.2].* 0.1 ); %JY: hard-coded
        xticks( [-3:1:3] .* 0.1 );
    end
    
    disp( dp_beta );
    disp( c_beta );
    
    betas.Dp{iPrm}   = [dp_beta, dp_se];
    betas.Crit{iPrm} = [c_beta, c_se];
    
end

end


