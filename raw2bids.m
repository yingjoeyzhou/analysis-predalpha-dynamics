%
% The script to arrange (old DCCN-) MEG data into bids format for project
% 3018041.02 (i.e., predalpha).
%
% Joey Zhou (Dec, 2023)

clearvars; clc; close all;

%% 
% =================================================
%           
%       Define the project directory
% 
% =================================================
proj_dir = '/Volumes/ExtDisk/DATA/3018041.02';
raw_dir  = fullfile( proj_dir, 'raw' );
bids_dir = fullfile( proj_dir, 'rawbids' );

% raw data from DCCN (by category)
meg_dir = fullfile( raw_dir, 'MegData' );
mri_dir = fullfile( raw_dir, 'MriData' );
beh_dir = fullfile( raw_dir, 'BehData' );
head_dir= fullfile( raw_dir, 'HeadData' );

% subjects present in the raw data
beh_mat  = dir( [beh_dir, filesep, 'Sub*.mat'] );
subjects = arrayfun(@(x) x.name(1:6), beh_mat, 'uniformoutput',false);
subjects = unique(subjects);


%% 
% =================================================
%           
%       Prep the folder structure
% 
% =================================================
for iSub = 1:numel(subjects)
    
    subject = subjects{iSub};
    
    % name of the subject foldder
    subj_dir = fullfile( bids_dir, subject );
    
    % check if the subject folder exists
    subj_dir_exist = isfolder( subj_dir );
    if subj_dir_exist
        fprintf('\nSubject-speficic rawbids folder (%s) already exists!', subj_dir);
        
        subj_meg_dir  = fullfile( subj_dir, 'meg' ); 
        subj_anat_dir = fullfile( subj_dir, 'anat' ); 
        subj_beh_dir  = fullfile( subj_dir, 'beh' );
    else
        fprintf('\nSubject-speficic rawbids folder (%s) about to be set up!', subj_dir);
        mkdir( subj_dir );
        
        subj_meg_dir  = fullfile( subj_dir, 'meg' ); mkdir( subj_meg_dir );
        subj_anat_dir = fullfile( subj_dir, 'anat' ); mkdir( subj_anat_dir );
        subj_beh_dir  = fullfile( subj_dir, 'beh' ); mkdir( subj_beh_dir );
    end
    
    %% ======================== MRI ==========================
    try
        
        % subject-specific MRI data
        subj_dicom_dir = sprintf('sub-0%s', subject(5:6));
        subj_dicom_dir = dir( [mri_dir, filesep, subj_dicom_dir, filesep, 'ses-mri01', filesep, '002-t1*'] );
        
        if isempty( subj_dicom_dir ) & strcmpi( subject, 'Sub236' )
            subj_dicom_dir = sprintf('sub-0%s', subject(5:6));
            subj_dicom_dir = fullfile( mri_dir, subj_dicom_dir );
        else
            subj_dicom_dir = fullfile( subj_dicom_dir.folder, subj_dicom_dir.name );
        end
        
        % from dicom_dir to anat_dir (in nifti.gz format)
        mri_dicom2niftigz( subj_dicom_dir, subject, subj_anat_dir );
        
        % label this subject 
        subj_mri_exist = true;
    catch
        subj_mri_exist = false;
    end
    
    %% ==================== Head Shape =====================
    try
        subj_hs_mat = dir( [head_dir, filesep, subject, '_headmodel.mat'] );
        load( fullfile(subj_hs_mat.folder, subj_hs_mat.name), 'vol' );
        
        hs_struct = vol.bnd.cfg.previous.previous.previous.callinfo.usercfg.headshape.headshape;
        pos = hs_struct.pos;
        for i = 1:numel(hs_struct.fid.label)
            eval(sprintf('pos_%s = hs_struct.fid.pos(%d,:);', hs_struct.fid.label{i}, i) );
        end
        save( fullfile(subj_anat_dir, sprintf('%s_headshape.mat',subject)), 'pos*' );
        
    catch
        fprintf('\nSomething went wrong when extracting the headshpae of %s..\n', subject);
        
        if subj_mri_exist
            
            % read in the mri
            dicomimg = dir( [subj_dicom_dir, filesep, '*.IMA'] );
            if isempty(dicomimg) & strcmpi(subject,'Sub236')
                dicomimg = dir( [subj_dicom_dir, filesep, 'anatomical.img'] );
            end
            subj_mri = ft_read_mri( fullfile(dicomimg(end).folder, dicomimg(end).name) );
            
            % results from manual alignment: transform_vox2ctf
            if strcmpi(subject, 'Sub234') %manually do this with fieldtrip
                cfg          = [];
                cfg.coordsys = 'ctf';
                cfg.method   = 'interactive';
                mri_realigned= ft_volumerealign(cfg, subj_mri);
                
                cfg.fiducial = mri_realigned.cfg.fiducial;
                
            else %read in the results
                cfg          = [];
                cfg.fiducial = vol.bnd.cfg.previous.previous.previous.fiducial;
                if isempty( cfg.fiducial )
                    cfg.fiducial = vol.bnd.cfg.previous.previous.previous.previous.fiducial;
                end
                cfg.method     = 'fiducial';
                cfg.coordsys   = 'ctf';
                mri_realigned  = ft_volumerealign(cfg, subj_mri);
            end
            
            fids = ft_transform_geometry(mri_realigned.transform, cfg.fiducial ); %fiducials here are in "mm" units
            labels = fields(fids);
            for i = 1:numel( labels )
                eval( sprintf('pos_%s = fids.%s / 10;', labels{i}, labels{i}) ); %divided by 10 to bring back to cm (headshapes are in "cm")
            end
            pos = pos_nas; %we have one datapoint (nas) for the polhemus headshape :)
            save( fullfile(subj_anat_dir, sprintf('%s_headshape.mat',subject)), 'pos*' );
        end
    end
    
    
    %{
    %% ======================== BEH =========================
    try 
        subj_beh_mat = dir( [beh_dir, filesep, subject, '_TaskMEG*.mat'] );
        copyfile( fullfile(subj_beh_mat.folder, subj_beh_mat.name), fullfile(subj_beh_dir, subj_beh_mat.name) );
    end
    
    
    %% ===================== MEG ==========================
    try 
        subj_ds_fname = sprintf('%s_3018041.02_*.ds', lower(subject));
        subj_ds_fname = dir( [meg_dir, filesep, subj_ds_fname] );
        
        dest_fname = fullfile(subj_meg_dir, subj_ds_fname.name);
        
        if ~exist(dest_fname, 'dir') %if the destination file does not exist
            copyfile( fullfile(subj_ds_fname.folder, subj_ds_fname.name), dest_fname );
        end
    end
    %}

    
end




%% ============= sub-function ============
function mri_dicom2niftigz( subj_dicom_dir, subject, subj_anat_dir )

ft_dir = '/Users/joeyzhou/Documents/GitHub/fieldtrip';
addpath(ft_dir);
ft_defaults;

cfg           = [];
cfg.parameter = 'anatomy';
cfg.filename  = fullfile( subj_anat_dir, sprintf('%s_T1w',subject) );
cfg.filetype  = 'nifti_gz';

if isempty( dir( [subj_anat_dir, filesep, '*.nii.gz']) )
    
    % load the dicom files
    dicomimg = dir( [subj_dicom_dir, filesep, '*.IMA'] );
    
    if isempty(dicomimg) & strcmpi( subject, 'Sub236' )
        dicomimg = dir( [subj_dicom_dir, filesep, 'anatomical.img'] );
    end
    subj_mri = ft_read_mri( fullfile(dicomimg(end).folder, dicomimg(end).name) );

    % write as a nifti.gz file in the rawbids subject-specific anat folder
    ft_volumewrite( cfg, subj_mri );

    % print progress
    fprintf('\n====== Successful in saving %s ========\n', cfg.filename);
    
    
else
    
    fprintf('\n====== File %s already there! ========\n', cfg.filename);

end

end
