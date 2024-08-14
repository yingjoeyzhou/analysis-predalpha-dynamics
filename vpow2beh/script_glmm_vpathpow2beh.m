

%% directories
figures_dir = '/Volumes/ExtDisk/analysis_DondersData/3018041.02/figures';
GLMM_dir    = '/Volumes/ExtDisk/analysis_DondersData/3018041.02/vpow2beh';

% Global parameters
vpath_thresh = 0;

% Models and states of interest
K_vec  = [8; 10]; %an (n_models,1) vector

ik_mat = [[1,2,3,6,7];... %an (n_models, n_networks) matrix
        [3,5,7,8,9]];

%% run GLMMs
%{ %
cnt = 0;
for K = K_vec'
    fprintf('\n\n\n Working on the %d-states model..', K);
    cnt = cnt + 1;
    for ik = ik_mat(cnt,:)
        fprintf('\n Zooming in to state #%d..\n', ik);
        glmm_vpathpow2beh_v2;
        %%glmm_vpathpow2beh_v1;
        %%compare_conditions_alphapow;
    end
end
%}



%% concatenate figures
close all;
cnt = 0;
for K = K_vec'
    fprintf('\n\n\n Working on the %d-states model..', K);
    cnt = cnt + 1;
    topos = {};
    conns = {};
    figDp = {};
    figCr = {};
    figCo = {};
    for ik = ik_mat(cnt,:)
        %figures showing HMM results
        topo_fname = sprintf('tde_hmm_%dstates_bestrun_alphapowertopo_state%d.png',K,ik); 
        conn_fname = sprintf('tde_hmm_%dstates_bestrun_alphanetwork_state%d.png',K,ik); 
        topos = horzcat( topos, fullfile(figures_dir,topo_fname) );
        conns = horzcat( conns, fullfile(figures_dir,conn_fname) );
        
        %figures showing GLMM results
        fig_betaD = sprintf('GLMM_betaDp_bestrun_%dth_of_%d_states.png', ik, K);
        %fig_betaC = sprintf('GLMM_betaCrit_bestrun_%dth_of_%d_states.png', ik, K);
        fig_betaC = sprintf('GLMM_betaC_bestrun_%dth_of_%d_states.png', ik, K);
        
        figDp = horzcat( figDp, fullfile(GLMM_dir, fig_betaD) );
        figCr = horzcat( figCr, fullfile(GLMM_dir, fig_betaC) );
        
        %figures showing condition comparison results
        %{
        fig_conds = sprintf('compcond_%s_bestrun_%dth_of_%d_states.png', pow_metric, ik, K);
        figCo = horzcat( figCo, fullfile(GLMM_dir, fig_conds) );
        %}
        
    end
    
    ncolumns = size(ik_mat,2);
    
    %{ %
    figure(cnt), hold on,
    montage( horzcat(figDp,figCr,topos), "Size",[3,ncolumns], 'BackgroundColor','white' );
    title( sprintf('%d-states model',K) );
    print(sprintf('GLMM_results_bestrun_of_%d_states.png', K), '-dpng','-r800');
    %}
    
    %{ 
    figure(cnt+1000), hold on,
    montage( horzcat(figCo,topos), "Size",[2,ncolumns], 'BackgroundColor','white' );
    title( sprintf('%d-states model',K) );
    print(sprintf('compcond_results_%s_bestrun_of_%d_states.png', pow_metric, K), '-dpng','-r800');
    %}
end