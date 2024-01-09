import glob
import os
import mne
import numpy as np
from osl import source_recon
from osl.source_recon.rhino import utils as rhino_utils

#%% deal with the input
import sys

# dearl with the input
print(sys.argv)
if sys.argv[0]=='':
    subject = input("Subject ID (e.g., Sub203):")
else:
    subject = sys.argv[1]
    print(sys.argv[1])

#%%Directories and files
source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")
PROJ_DIR  = "/ohba/pi/knobre/joeyzhou/3018041.02"
RAW_DIR   = f"{PROJ_DIR}/rawbids"
DERIV_DIR = f"{PROJ_DIR}/derivatives"
SRC_DIR   = f"{DERIV_DIR}/src"

#%% ============ sub-function ==============
def define_input_and_output( subject ):
    '''Define the fnames of inputs, given a subject.
    '''
    #Subject-specific .ds folder (inputs)
    SUBJ_MEG_CLEANED = glob.glob( f"{DERIV_DIR}/cleaned/{subject}_cleaned_raw/*cleaned_preproc_raw.fif" )
    SUBJ_MRI_RAW     = glob.glob( f"{RAW_DIR}/{subject}/anat/{subject}_T1w.nii.gz" )
    
    return SUBJ_MEG_CLEANED[0], SUBJ_MRI_RAW[0]


#%% ============ sub-function ============
def break_array_into_chunks(array, k, out_dir):
    m, n = array.shape
    subarray_size = n // k  # Determine the approximate size of each subarray
    start_col = 0

    out_fnames = []

    if os.path.isdir( out_dir )==False:
        os.mkdir(out_dir)
    
    for i in range(k - 1):
        end_col = start_col + subarray_size
        
        # Create subarray from start_col to end_col
        subarray = array[:, start_col:end_col]

        # Save the chunk
        fname = f"{out_dir}/beamformed_raw_chunks{i+1}.npy"
        np.save(fname, subarray)
        out_fnames.append(fname)
        
        # Update the start column
        start_col = end_col

        # Show progress
        print(f"Finished saving {fname}")
    
    # Last chunk might be slightly bigger to cover remaining columns
    last_subarray = array[:, start_col:]

    # Save the last chunk 
    fname = f"{out_dir}/beamformed_raw_chunks{k}.npy"
    np.save(fname, last_subarray)
    out_fnames.append(fname)  

    print(f"Finished saving {fname}")  
    
    return out_fnames

#%% =========== sub-function ==========
def transform_recon_mri( subjects_dir, subject, spatial_resolution=None, reference_brain="mni"):
    '''This function is PART OF a direct copy of `osl.source_recon.beamforming.transform_recon_timeseries`.
    The only difference is that we avoid holding `recon_timeseries` in the machine's working memory, 
    so that FSL does not complain about memory shortage when called. 
    '''

    from mne import read_forward_solution
    from osl.utils.logger import log_or_print   
    import os
    import os.path as op

    rhino_files = rhino_utils.get_rhino_files(subjects_dir, subject)
    surfaces_filenames = rhino_files["surf"]
    coreg_filenames = rhino_files["coreg"]

    # -------------------------------------------------------------------------------------
    # Get hold of coords of points reconstructed to
    #
    # Note, MNE forward model is done in head space in metres. RHINO does everything in mm.
    fwd = read_forward_solution(rhino_files["fwd_model"])
    vs = fwd["src"][0]
    recon_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    # ----------------------------
    if spatial_resolution is None:
        # Estimate gridstep from forward model
        rr = fwd["src"][0]["rr"]

        store = []
        for ii in range(rr.shape[0]):
            store.append(np.sqrt(np.sum(np.square(rr[ii, :] - rr[0, :]))))
        store = np.asarray(store)
        spatial_resolution = int(np.round(np.min(store[np.where(store > 0)]) * 1000))

    spatial_resolution = int(spatial_resolution)
    log_or_print(f"spatial_resolution = {spatial_resolution} mm")

    if reference_brain == "mni":
        # Reference is mni stdbrain

        # Convert recon_coords_head from head to mni space, head_mri_t_file xform is to unscaled MRI
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_mri = rhino_utils.xform_points(head_mri_t["trans"], recon_coords_head.T).T

        # mni_mri_t_file xform is to unscaled MRI
        mni_mri_t = rhino_utils.read_trans(surfaces_filenames["mni_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(np.linalg.inv(mni_mri_t["trans"]), recon_coords_mri.T).T

        reference_brain = os.environ["FSLDIR"] + "/data/standard/MNI152_T1_1mm_brain.nii.gz"

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = op.join(coreg_filenames["basedir"], "MNI152_T1_{}mm_brain.nii.gz".format(spatial_resolution))

    elif reference_brain == "unscaled_mri":
        # Reference is unscaled smri

        # convert recon_coords_head from head to mri space
        head_mri_t = rhino_utils.read_trans(coreg_filenames["head_mri_t_file"])
        recon_coords_out = rhino_utils.xform_points(head_mri_t["trans"], recon_coords_head.T).T

        reference_brain = surfaces_filenames["smri_file"]

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(".nii.gz", "_{}mm.nii.gz".format(spatial_resolution))

    elif reference_brain == "mri":
        # Reference is scaled smri

        # Convert recon_coords_head from head to mri space
        head_scaledmri_t = rhino_utils.read_trans(coreg_filenames["head_scaledmri_t_file"])
        recon_coords_out = rhino_utils.xform_points(head_scaledmri_t["trans"], recon_coords_head.T).T

        reference_brain = coreg_filenames["smri_file"]

        # Sample reference_brain to the desired resolution
        reference_brain_resampled = reference_brain.replace(".nii.gz", "_{}mm.nii.gz".format(spatial_resolution))

    else:
        ValueError("Invalid out_space, should be mni or mri or scaledmri")

    # ---------------------------------------------------------------------
    # Get coordinates from reference brain at resolution spatial_resolution

    # Create std brain of the required resolution
    rhino_utils.system_call("flirt -in {} -ref {} -out {} -applyisoxfm {}".format(reference_brain, reference_brain, reference_brain_resampled, spatial_resolution))

    coords_out, vals = rhino_utils.niimask2mmpointcloud(reference_brain_resampled)


    #% variables for later use
    outputs_recon_mri = {}
    outputs_recon_mri['coords_out']                = coords_out
    outputs_recon_mri['reference_brain_resampled'] = reference_brain_resampled
    outputs_recon_mri['recon_coords_out']          = recon_coords_out          
    outputs_recon_mri['spatial_resolution']        = spatial_resolution
    outputs_recon_mri['reference_brain']           = reference_brain

    return outputs_recon_mri


#%% =========== sub-function =============
def transform_recon_timeseries_stepwise( recon_timeseries, outputs_recon_mri ):
    '''This function is a direct copy of `osl.source_recon.beamforming.transform_recon_timeseries`.
    The only difference is that we avoid holding `recon_timeseries` in the machine's working memory, 
    so that FSL does not complain about memory shortage when called.
    '''

    # THIS WAS DEFINED BY `transform_recon_mri`
    coords_out = outputs_recon_mri['coords_out'] 
    reference_brain_resampled = outputs_recon_mri['reference_brain_resampled']
    recon_coords_out = outputs_recon_mri['recon_coords_out'] 
    spatial_resolution = outputs_recon_mri['spatial_resolution']
    reference_brain = outputs_recon_mri['reference_brain']

    # --------------------------------------------------------------
    # For each mni_coords_out find nearest coord in recon_coords_out

    recon_timeseries_out = np.zeros(np.insert(recon_timeseries.shape[1:], 0, coords_out.shape[1]))

    recon_indices = np.zeros([coords_out.shape[1]])

    for cc in range(coords_out.shape[1]):
        recon_index, dist = rhino_utils._closest_node(coords_out[:, cc], recon_coords_out)

        if dist < spatial_resolution:
            recon_timeseries_out[cc, :] = recon_timeseries[recon_index, ...]
            recon_indices[cc] = recon_index

    return recon_timeseries_out, reference_brain_resampled, coords_out, recon_indices
    

#%% =========== key function ============
def parcellate_raw(src_dir, subject, preproc_file, parcellation_file, method, orthogonalisation, 
                   spatial_resolution=None, reference_brain="mni", extra_chans="stim"):
    '''This is written based on `osl.source_recon.wrappers.parcellate`, with improved efficiency in handling
    long continuous raw data.
    '''
    import pickle
    from osl.source_recon import parcellation 
    from osl.report import src_report
    
    import logging
    logger = logging.getLogger(__name__) 

    # Prepare the coords 
    # RUN THIS FIRST TO AVOID MEMORY SHORTAGE AFTER READING IN THE RAW MEG DATA)
    outputs_recon_mri = transform_recon_mri( src_dir, subject, spatial_resolution=None, reference_brain="mni")


    # Prepare the coords for parcellation
    parcellation_asmatrix = source_recon.parcellation.parcellation._resample_parcellation(
                                            parcellation_file,
                                            voxel_coords=outputs_recon_mri['coords_out'],
                                            working_dir=f"{src_dir}/{subject}/parc")


    # Get settings passed to the beamform wrapper
    report_data = pickle.load(open(f"{src_dir}/{subject}/report_data.pkl", "rb"))
    freq_range = report_data.pop("freq_range")
    chantypes = report_data.pop("chantypes")
    if isinstance(chantypes, str):
        chantypes = [chantypes]

    # Load data
    raw = mne.io.read_raw_fif(preproc_file, preload=True)    

    # Pick channels
    chantype_data = raw.copy().pick(chantypes)    

    # Load beamforming filter and apply
    filters = source_recon.beamforming.load_lcmv(src_dir, subject)
    bf_data = source_recon.beamforming.apply_lcmv(chantype_data, filters)

    # Release some memory
    del raw, chantype_data, filters

    # Beamformed data in MNI space
    bf_data = bf_data.data

    # Reconstruct the source-level timeseries data and parcellate
    try:
        bf_data_mni, _, coords_mni, _ = transform_recon_timeseries_stepwise(
                                        recon_timeseries=bf_data, 
                                        outputs_recon_mri=outputs_recon_mri,
                                        )
        
        '''
        # ====== in `osl.source_recon.wrappers.parcellate` =======
        bf_data_mni, _, coords_mni, _ = beamforming.transform_recon_timeseries(
            subjects_dir=src_dir,
            subject=subject,
            recon_timeseries=bf_data,
            spatial_resolution=spatial_resolution,
            reference_brain=reference_brain,
        )
        # ====== in `osl.source_recon.wrappers.parcellate` =======
        '''

        # free up some memory
        del bf_data 

        # Parcellation
        logger.info(f"using file {parcellation_file}")
        parcel_data, _, _ = parcellation.parcellate_timeseries(
            parcellation_file,
            voxel_timeseries=bf_data_mni,
            voxel_coords=coords_mni,
            method=method,
            working_dir=f"{src_dir}/{subject}/parc",
        )

        # free up some memory
        del bf_data_mni

    except MemoryError:

        # break the huge array into chunks and save to disk
        num_chunks   = 30
        chunk_dir    = f"{src_dir}/{subject}/temp"
        chunk_fnames = break_array_into_chunks(bf_data, num_chunks, chunk_dir)

        # free up some memory
        del bf_data        

        logger.info(f"using file {parcellation_file}")

        # parcellate the raw data in chunks
        parcel_data = []
        for i, chunk_fname in enumerate(chunk_fnames):
            recon_dat = np.load( chunk_fname )
            bf_dat_mni, _, coords_mni, _ = transform_recon_timeseries_stepwise(
                                        recon_timeseries=recon_dat,
                                        outputs_recon_mri=outputs_recon_mri,
                                        )
            parcel_dat, _, _ = source_recon.parcellation.parcellation._get_parcel_timeseries(
                                    voxel_timeseries=bf_dat_mni, 
                                    parcellation_asmatrix=parcellation_asmatrix, 
                                    method=method)
            del recon_dat, bf_dat_mni
            parcel_data.append( parcel_dat )
            print(f"Done parcellating chunk #{i}!")

        # free up some memory
        del parcel_dat

        # make it into one big array
        parcel_data = np.hstack(parcel_data)


    # For debugging
    np.save(f"{src_dir}/{subject}/parc/parcel_data.npy", parcel_data)


    # Orthogonalisation
    if orthogonalisation not in [None, "symmetric", "none", "None"]:
        raise NotImplementedError(orthogonalisation)

    if orthogonalisation == "symmetric":
        logger.info(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.symmetric_orthogonalise(parcel_data, maintain_magnitudes=True)

    
    # Save parcellated data as a MNE Raw object
    raw = mne.io.read_raw_fif(preproc_file, preload=True) 
    parc_fif_file = f"{src_dir}/{subject}/parc/parc-raw.fif"
    logger.info(f"saving {parc_fif_file}")
    parc_raw = parcellation.convert2mne_raw(parcel_data, raw, extra_chans=extra_chans)
    parc_raw.save(parc_fif_file, overwrite=True)

    # Save plots
    parc_psd_plot = f"{subject}/parc/psd.png"
    parcellation.plot_psd(
        parcel_data,
        fs=raw.info["sfreq"],
        freq_range=freq_range,
        parcellation_file=parcellation_file,
        filename=f"{src_dir}/{parc_psd_plot}",
    )
    parc_corr_plot = f"{subject}/parc/corr.png"
    parcellation.plot_correlation(parcel_data, filename=f"{src_dir}/{parc_corr_plot}")

    # Save info for the report
    n_parcels = parcel_data.shape[0]
    n_samples = parcel_data.shape[1]
    if parcel_data.ndim == 3:
        n_epochs = parcel_data.shape[2]
    else:
        n_epochs = None
    src_report.add_to_data(
        f"{src_dir}/{subject}/report_data.pkl",
        {
            "parcellate": True,
            "parcellation_file": parcellation_file,
            "method": method,
            "orthogonalisation": orthogonalisation,
            "parc_fif_file": str(parc_fif_file),
            "n_samples": n_samples,
            "n_parcels": n_parcels,
            "n_epochs": n_epochs,
            "parc_psd_plot": parc_psd_plot,
            "parc_corr_plot": parc_corr_plot,
        },
    )


#%%
# raw MEG file
preproc_file, _ = define_input_and_output( subject )

# do the work
parcellate_raw(src_dir=SRC_DIR, 
                subject=subject, 
                preproc_file=preproc_file, 
                parcellation_file="/ohba/pi/knobre/joeyzhou/3018041.02/analysis/preproc/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
                method="spatial_basis", 
                orthogonalisation="symmetric", 
                spatial_resolution=None, 
                reference_brain="mni", 
                extra_chans="stim")