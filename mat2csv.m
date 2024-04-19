% Transform .mat files into .csv files, so that python can read.
%
% JY (Apr, 2024)

clearvars; close all; clc;

%% 
% =================================================
%           
%       Define the project directory
% 
% =================================================
proj_dir = '/Volumes/ExtDisk/DATA/3018041.02';
bids_dir = fullfile( proj_dir, 'rawbids' );
subj_dir = dir([bids_dir,filesep,'Sub*']);
subjects = arrayfun( @(x) x.name, subj_dir, 'uniformoutput',false); %output={'Sub202','Sub203' xx}


%%
% =================================================
%
%       Define each trial
%
% =================================================
for ii = 1:numel(subjects)
    subject = subjects{ii};
    beh_dir = fullfile( subj_dir(ii).folder, subject, 'beh' );
    
    beh_file = dir([beh_dir, filesep, [subject,'_taskMEG*.mat']]);
    par_file = dir([beh_dir, filesep, [subject,'_paramMEG.mat']]);
    
    assert( numel(beh_file)==1 );
    assert( numel(par_file)==1 );
    
    data = load( fullfile(beh_file.folder, beh_file.name) );
    par  = load( fullfile(par_file.folder, par_file.name) );
    
    %main task
    tIdx = [];
    bIdx = [];
    corr = [];
    rt   = [];
    stimori  = [];
    presence = [];
    primetype= [];
    taskname = [];
    for b = 1:numel(data.mainTrl)
        
        %priming task
        n = par.proc.nPrimeTrialsPerBlock;
        tIdx = [tIdx, 1:n];
        bIdx = [bIdx, repmat(b,[1,n])];
        corr = [corr, [data.primeTrl{b}.resp(:).correct] ];
        rt   = [rt, [data.primeTrl{b}.resp(:).rt] ];
        presence  = [presence, data.primeTrl{b}.type ];
        taskname  = [taskname; repmat({'prime'},[n,1])];
        
        %main task
        n = par.proc.nMainTrialsPerBlock;
        tIdx = [tIdx, (1:n)+par.proc.nPrimeTrialsPerBlock ];
        bIdx = [bIdx, repmat(b,[1,n])];
        corr = [corr, [data.mainTrl{b}.resp(:).correct] ];
        rt   = [rt, [data.mainTrl{b}.resp(:).rt] ];
        presence  = [presence, data.mainTrl{b}.type ];
        taskname  = [taskname; repmat({'main'},[n,1])];
        
        %shared
        n = par.proc.nMainTrialsPerBlock + par.proc.nPrimeTrialsPerBlock;
        stimori   = [stimori, repmat(par.proc.stimOri(b), [1,n]) ];
        primetype = [primetype, repmat(par.proc.primeType(b), [1,n]) ];
    end
    X = [tIdx',bIdx',corr',rt',stimori',presence',primetype'];
    T = array2table( X, 'VariableNames',{'tIdx','bIdx','correct','rt','stimori','presence','primetype'} );
    T.taskname = taskname;
    T.trig_stimon = repmat( par.triggers.main.stimOn, [size(X,1),1] );
    T.trig_respon = repmat( par.triggers.main.promptOn, [size(X,1),1] );
    
    %a very quick check
    assert( mean(T.presence)==0.5 );
    
    %save the trial-labeling
    out_fname = sprintf('%s_maintask_metadata.csv',subject);
    out_fname = fullfile( beh_dir, out_fname );
    writetable( T, out_fname );
    
end

