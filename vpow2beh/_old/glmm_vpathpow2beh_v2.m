%
% Run GLMM to see if pre-stim alpha power predicts d' or criterion.
%
% In the GLMM, vpath was thresholded and categorized into ON vs. OFF.
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

stat = struct('Fstat',[], 'Pval',[]);
beta = struct('Dp',[], 'C',[]);

primetype = {'cons','lib'};

for iT = 1:numel(tvec)
    
    pname = powtnames{iT};
    vname = strrep( pname, 'pow','vpath');
    alpha = data.(pname);
    vpath = data.(vname);
    
    %normalize within subject
    for iSub = 1:n
        sel = (vec_subject==iSub);
        alpha(sel) = zscore(log(alpha(sel)));
    end
    
    %discretize vprob
    vpath = ( vpath>vpath_thresh );
    
    %loop through the two prime types
    coef_table = cell(2,1);
    cov_table  = cell(2,1);
    
    t = array2table( horzcat(X,alpha,vpath), ...
                        'VariableNames',{'y', 'S', 'subject', 'primetype', 'P', 'V'});
    
    % Test the effect of alphapow on y (i.e., probability report yes) while
    % controling for stimulus presence and test for interaction effects.
    % Model random subject intercepts and slopes.
    % Intercept is modeled implicitly both at the group and subject level.
    t.S         = categorical( t.S ); %dummy-code
    t.V         = categorical( t.V ); %dummy-code ON vs. OFF
    t.subject   = categorical( t.subject ); %dummy-code
    t.primetype = categorical( t.primetype ); %dummy-code
    m = fitglme( t, 'y ~ 1+S*P*V + (1+S | subject:primetype)',...
            'Distribution', 'Binomial',...
            'Link', 'probit');
    
    % Extract variable of interests
    [dp_beta,c_beta,sout] = get_betas_and_stats(m);

    % Concatenate
    beta.Dp = [ beta.Dp; dp_beta ]; %beta.Dp is an nTimes-by-2(ON/OFF) array
    beta.C = [ beta.C; c_beta ]; %beta.C is an nTimes-by-2(ON/OFF) array
    
    stat.Pval = [ stat.Pval; sout.Pvec ]; %stat.Pvals is an nTimes-by-nContrasts array
    stat.Fstat = [ stat.Fstat; sout.Fvec ]; %stat.Fvals is an nTimes-by-nContrasts array
    
end

%% Do the plots: one for criterion one for dprime
close all;
colors = struct('ON',[178,24,43]./255,...
                'OFF',[33,102,172]./255);
for iV = 1:2
    figure, hold on,
    set(gcf,'units','centimeters','position',[0 0 6 4]);
    switch iV
        case 1, vname='Dp'; yname='\beta_d_'''; iON=1; iOFF=2; iInt=3;
        case 2, vname='C'; yname='\beta_c'; iON=4; iOFF=5; iInt=6;
    end

    %plot zero-line
    yline(0, 'k--');

    %plot the ON
    lineON = plot( tvec, beta.(vname)(:,1), '-',...
                    'Color',colors.ON,'linewidth',1.5);

    %plot the OFF
    lineOFF = plot( tvec, beta.(vname)(:,2), '-', ...
        'linewidth',1.5, 'Color',colors.OFF);

    %Se yscale
    ymax = max( abs(beta.(vname)(:)) );
    ylim( [-1,1].*ymax.*1.5 );
    YLim = ylim();
    ylabel( yname, 'FontSize',10 );

    %adjust the x-scale
    xlabel( 'Time (s)', 'FontSize',10);
    if tvec(end)<0.1
        xticks( -1:0.25:0 );
    else
        xticks( -1:0.5:1 );
    end
    
    % ================ Plot significance =============
    iSigLine = 0;
    
    %Plot significant interaction between P and V
    if any(stat.Pval(:,iOFF)<0.05)
        iSigLine = iSigLine + 1;
        ysig = YLim(1) + diff(YLim)*0.025*iSigLine;
        t_masked = tvec( stat.Pval(:,iInt)<0.05 );
        midx = find( diff(t_masked)>(tvec(2)-tvec(1)) );
        midx = [midx, numel(t_masked), numel(t_masked)];
        for id = 1:(numel(midx)-1)
            if id==1
                curridx = 1:(midx(1));
            else
                curridx = (midx(id-1)+1):midx(id);
            end

            plot( t_masked(curridx), ones(size(curridx)).*ysig, 'k-', 'linewidth',2);
        end
    end
    
    %Plot significant ON
    if any(stat.Pval(:,iON)<0.05)
        iSigLine = iSigLine + 1;
        ysig = YLim(1) + diff(YLim)*0.025*iSigLine;
        time_masked = tvec(stat.Pval(:,iON)<0.05);
        midx = find( diff(time_masked)>(tvec(2)-tvec(1)) );
        midx = [midx, numel(time_masked), numel(time_masked)];
        for id = 1:(numel(midx)-1)
            if id==1
                curridx = 1:(midx(1));
            else
                curridx = (midx(id-1)+1):midx(id);
            end
            plot( time_masked(curridx), ones(size(curridx)).*ysig, '-', ...
                    'Color',colors.ON, 'linewidth',2);
    % % % %         plot( time_masked(curridx), var_masked(curridx), '-', ...
    % % % %             'Color',colors.ON, 'linewidth',4);
        end
    end
    
    %Plot significant OFF
    if any(stat.Pval(:,iOFF)<0.05)
        iSigLine = iSigLine + 1;
        ysig = YLim(1) + diff(YLim)*0.025*iSigLine;
        time_masked = tvec(stat.Pval(:,iOFF)<0.05);
        midx = find( diff(time_masked)>(tvec(2)-tvec(1)) );
        midx = [midx, numel(time_masked), numel(time_masked)];
        for id = 1:(numel(midx)-1)
            if id==1
                curridx = 1:(midx(1));
            else
                curridx = (midx(id-1)+1):midx(id);
            end
            plot( time_masked(curridx), ones(size(curridx)).*ysig, '-', ...
                    'Color',colors.OFF, 'linewidth',2);
        end
    end


    
    %Set figure
    set(gca,'TickDir','out', 'FontSize',8);
    %xlim( [-0.8, 0.8] );
    xlim( [tvec(1)-0.05, tvec(end)+0.05] );
    
    %Legend
    %{
    legend([lineON,lineOFF],{'V=1','V=0'},...
        'NumColumns',2,'Location','northwest','FontSize',8);
    legend boxoff
    %}
    
    %Save the figure
    print(sprintf('GLMM_beta%s_bestrun_%dth_of_%d_states.png', vname, ik, K), '-dpng','-r400');
    
end

%% Save the results
save(sprintf('GLMM_pow_x_vpath_bestrun_%dth_of_%d_states.mat', ik, K));



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

%% ============ sub-function =============
function [dp_beta, c_beta, statout] = get_betas_and_stats(m)

    H = zeros(6, numel(m.CoefficientNames) );
    dp_beta = nan(1,2);
    c_beta  = nan(1,2);
    
    % d' modulation when the network is ON
    idx_P2d = ismember( m.CoefficientNames, {'S_1:P', 'S_1:P:V_1'}); 
    H(1,idx_P2d) = 1;
    dp_beta(1) = H(1,:) * m.Coefficients.Estimate;
    
    % d' modulation when the network is OFF
    idx_P2d = ismember( m.CoefficientNames, {'S_1:P'} );
    H(2,idx_P2d) = 1;
    dp_beta(2) = H(2,:) * m.Coefficients.Estimate;
    
    % Update the contrast to test d' modulation network ON vs. OFF
    H(3,:) = H(1,:) - H(2,:);
    
    % criterion modulation when network is ON
    idx_P2c = ismember( m.CoefficientNames, {'P', 'P:V_1'});
    H(4,idx_P2c) = -1;
    idx_P2c = ismember( m.CoefficientNames, {'S_1:P', 'S_1:P:V_1'});
    H(4,idx_P2c) = -1/2;
    c_beta(1) = H(4,:) * m.Coefficients.Estimate;
    
    % critrion modulation when netwrk is OFF
    idx_P2c = ismember( m.CoefficientNames, {'P'});
    H(5,idx_P2c) = -1;
    idx_P2c = ismember( m.CoefficientNames, {'S_1:P'});
    H(5,idx_P2c) = -1/2;
    c_beta(2) = H(5,:) * m.Coefficients.Estimate;
    
    % Update the contrast to test d' modulation network ON vs. OFF
    H(6,:) = H(4,:) - H(5,:);
    
    % Run stats on the specified contrasts
    Pvec = nan(1,size(H,1));
    Fvec = nan(1,size(H,1));
    for ii = 1:size(H,1)
        [Pvec(ii), Fvec(ii)] = m.coefTest( H(ii,:) );
    end
    
    % Contrast names: JY Hard coded
    contrastnames = {'dprime-ON', 'dprime-OFF', 'dprime-ON-vs-OFF', ...
                    'criterion-ON', 'criterion-OFF', 'criterion-ON-vs-OFF'};
                
    statout = struct('Pvec',Pvec,...
                    'Fvec',Fvec,...
                    'contrastnames',contrastnames);
                
end
    
