%
% Plot model prediction.
%
% JY (June 2024)


% COLOR-CODES
cmaps = struct('cons',[123,50,148]./255,...
               'lib', [0,136,55]./255,...
               'ON',[178,24,43]./255,...
               'OFF',[33,102,172]./255);


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

% Which time point 
if ~exist('toi2plot', 'var')
    toi2plot = input('Please specify the timepoint of interest (e.g., -0.25).\n', 's');
    toi2plot = str2double(toi2plot);
end

%% filename and directories
% directory and fnames
CODE_DIR = '/Volumes/ExtDisk/analysis_DondersData/3018041.02/vpow2beh';
CSV_DIR  = fullfile( CODE_DIR, 'csv_stateprob_parc9975' );
addpath( genpath(CODE_DIR) );
addpath('/Users/joeyzhou/Documents/MATLAB/shadedErrorBar');

% csv filename 
csv_fname = sprintf('group_pow_and_vpath_%dstatesmodel_state%d.csv', K, ik);
mat_fname = sprintf('GLMM_pow_x_vpath_bestrun_%dth_of_%d_states.mat', ik, K);
disp( mat_fname );

% load the GLMM results file
results = load( mat_fname );


%% repeat the fit
[m,t] = fitGLMM_given_toi( toi2plot, results );


%% generate new prediction
primecodes = struct('cons',1, 'lib',0); %JY: hard-coded conservative is coded as 1
pm_fields  = fields(primecodes);
vpathcodes = struct('on',1, 'off',0); %JY: hard-coded ON state is coded as 1
v_fields   = fields(vpathcodes);

subjects = double( unique( t.subject ));
primings = [0, 1]; %JY: hard-coded
powervec = linspace(-3,3,1000);
vpathvec = [0, 1];
stimvec  = [0, 1];

[subject, primetype, P, S, V] = ndgrid( subjects, primings, powervec, stimvec, vpathvec );
tnew = array2table( horzcat(S(:), subject(:), primetype(:), P(:), V(:)), ...
                    'VariableNames',{'S', 'subject', 'primetype', 'P', 'V'});
tnew.S         = categorical( tnew.S ); %dummy-code
tnew.V         = categorical( tnew.V ); %dummy-code ON vs. OFF
tnew.subject   = categorical( tnew.subject ); %dummy-code
tnew.primetype = categorical( tnew.primetype ); %dummy-code

nans  = nan( numel(powervec), 2, numel(subjects) );
zmat  = [];
zmat.cons = struct('on',nans, 'off',nans);
zmat.lib  = struct('on',nans, 'off',nans);

for iV = 1:2
    
    
    vfield = v_fields{ structfun( @(x) x==vpathvec(iV), vpathcodes, 'uniformoutput',true) };
    

    for iS = 1:2
        
        sel0 = (tnew.V==categorical(vpathvec(iV))) & ( tnew.S==categorical(stimvec(iS)) );
        
        for iSub = 1:numel(subjects)
            for iPm = 1:numel(primings)
                sel = (tnew.subject==categorical(subject(iSub))) & ( tnew.primetype==categorical(primings(iPm)) );
                tIN = tnew(sel&sel0, :);
                ypred = m.predict(tIN); %Hit or FA rate (probability of Y=1 response)
                pmfield = pm_fields{ structfun( @(x) x==primings(iPm), vpathcodes, 'uniformoutput',true) };
                
                zmat.(pmfield).(vfield)(:,iS,iSub) = norminv( ypred );
                
            end
        end
        
    end
end




%% Do the plot: one panel for ON, the other for OFF
close all;

for iVar = 1:2
    figure, hold on,
    set(gcf,'units','centimeters','position',[0 0 8 4]);
    switch iVar
        case 1, vname='Dp'; yname='d'''; ylim2use=[0.5, 2.5];
        case 2, vname='Crit'; yname='c'; ylim2use=[-0.5, 1.5];
    end
    
    for iV = 1:2 %For ON and OFF states separately

        subplot(1,2,iV), hold on,

        vfield = v_fields{ structfun( @(x) x==vpathvec(iV), vpathcodes, 'uniformoutput',true) };

        % JY: hard-coded
        ipres = stimvec==1;
        iabs  = stimvec==0;

        % Two lines for the two priming conditions
        for iPm = 1:2
            pmfield = pm_fields{ structfun( @(x) x==primings(iPm), vpathcodes, 'uniformoutput',true) };    

            %dmat is an n(z-scored)power-by-nsubjects matrix of d'
            dmat = squeeze( zmat.(pmfield).(vfield)(:,ipres,:)-zmat.(pmfield).(vfield)(:,iabs,:) );
            
            %cmat is an n(z-scored)power-by-nsubjects matrix of criterion
            cmat = squeeze( (-0.5) .* sum(zmat.(pmfield).(vfield),2) );
            
            switch vname
                case 'Dp', ymat = dmat;
                case 'Crit', ymat = cmat;
            end
            m_var = nanmean(ymat,2); 
            e_var = std(ymat,[],2)./sqrt(size(ymat,2)); %between-subject SE

            %plot model prediction
            plot( powervec, m_var, 'linewidth',1.5, 'Color',cmaps.(pmfield));
            shadedErrorBar( powervec, m_var, e_var, ...
                'lineProps',{'Color',cmaps.(pmfield)},...
                'transparent',1 );

        end

        % Make the axes of different colors
        set(gca,'TickDir','out',...
            'XColor',cmaps.(upper(vfield)),...
            'YColor',cmaps.(upper(vfield)),...
            'FontSize',8);
        
        % Set ylim and xlim
        ylim( ylim2use ); %ylim( [min(ymat(:)), max(ymat(:))] + [-0.1, 0.1] );
        xlim( [min(powervec), max(powervec)] + [-0.05,0.05] );
        if iV==1; ylabel(yname,'FontSize',10, 'Color','k'); end
        xlabel('zscore(log\alpha)','FontSize',10);

    end
    
    figfname = sprintf('GLMM_modelpred%s_t%s_bestrun_%dth_of_%d_states.png', vname, num2str(toi2plot*1000), ik, K);
    print(figfname, '-dpng','-r400');
    
end




%% =============== sub-function ================
function [m,t] = fitGLMM_given_toi( toi2plot, results )
% This needs to be consistent with the corresponding `glmm_vpathpow2beh`
% function used to obtain the results.
% 
% JY (June 2024)

[~,iT] = min( abs(results.tvec - toi2plot) );
pname = results.powtnames{iT};
vname = strrep( pname, 'pow','vpath');
alpha = results.data.(pname);
vpath = results.data.(vname);
    
%normalize within subject
for iSub = 1:results.n
    sel = (results.vec_subject==iSub);
    alpha(sel) = zscore(log(alpha(sel)));
end
    
%discretize vprob
vpath = ( vpath>results.vpath_thresh );

%input table
t = array2table( horzcat(results.X,alpha,vpath), ...
                    'VariableNames',{'y', 'S', 'subject', 'primetype', 'P', 'V'});
%run the fit
t.S         = categorical( t.S ); %dummy-code
t.V         = categorical( t.V ); %dummy-code ON vs. OFF
t.subject   = categorical( t.subject ); %dummy-code
t.primetype = categorical( t.primetype ); %dummy-code
m = fitglme( t, char( results.m.Formula ),...
        'Distribution', 'Binomial',...
        'Link', 'probit');
end