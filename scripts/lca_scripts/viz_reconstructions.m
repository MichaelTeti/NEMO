function viz_reconstructions(openpv_path, checkpoint_dir, save_dir, rec_key, input_key)
    %{
        Reads in inputs and reconstructions from a checkpoint and writes them out.
         
        Args:
            openpv_path: The path to OpenPV/mlab/util.
            checkpoint_dir: The path to the checkpoint directory where the input and 
                recon .pvp files are. 
            save_dir: The directory where the inputs and recons will be saved in separate
                subdirectories.
            rec_key: The key used to find the recon .pvp files (e.g. Frame*Recon_A.pvp). 
            input_key: The key used to find the input .pvp files (e.g. Frame*_A.pvp).
    %}

    addpath(openpv_path);

    % check if the checkpoint dir given exists
    if ~exist(checkpoint_dir, 'dir')
       printf('Directory given to viz_reconstructions does not exist.')
       return
    end

    % add forward slash to checkpoint dir if not there
    if checkpoint_dir(end) ~= '/'
      checkpoint_dir = strcat(checkpoint_dir, '/');
    end

    % add forward slash to save dir if not there
    if save_dir(end) ~= '/'
      save_dir = strcat(save_dir, '/');
    end

    % if save_dir doesn't exit, make it
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % make different dirs to save inputs and recons and diffs
    rec_dir = strcat(save_dir, 'Recons/');
    input_dir = strcat(save_dir, 'Inputs/');
    diff_dir = strcat(save_dir, 'Diffs/');

    if ~exist(rec_dir, 'dir')
        mkdir(rec_dir);
    end

    if ~exist(input_dir, 'dir')
        mkdir(input_dir);
    end
    
    if ~exist(diff_dir, 'dir')
        mkdir(diff_dir);
    end

    % find the fpaths with recons and inputs in checkpoint dir
    rec_fpaths = dir(strcat(checkpoint_dir, rec_key));
    input_fpaths = dir(strcat(checkpoint_dir, input_key));
    n_inputs = size(readpvpfile(strcat(checkpoint_dir, rec_fpaths(1, 1).name)), 1);
    n_fpaths = numel(rec_fpaths);

    for batch_num = 1:n_inputs
        rec_fpath_save = strcat(rec_dir, strcat(int2str(batch_num), '.gif'));
        input_fpath_save = strcat(input_dir, strcat(int2str(batch_num), '.gif'));
        diff_fpath_save = strcat(diff_dir, strcat(int2str(batch_num), '.gif'));
    
        for frame_num = 1:n_fpaths
            % read in the inputs and recon for this batch sample and video frame
            rec_fpath = strcat(checkpoint_dir, rec_fpaths(frame_num, 1).name);
            input_fpath = strcat(checkpoint_dir, input_fpaths(frame_num, 1).name);
            rec = readpvpfile(rec_fpath);
            inputs = readpvpfile(input_fpath);
            rec = rec{batch_num, 1}.values;
            inputs = inputs{batch_num, 1}.values;

            % go from x, y to y, x to save as images
            rec = transpose(rec);
            inputs = transpose(inputs);
            
            % get the difference
            diff = inputs - rec;
            
            % make a placeholder to aggregate frames from each batch sample
            % so we can scale them by statistics of all frames for that sample
            if frame_num == 1
                height = size(rec, 1);
                width = size(rec, 2);
                recs_agg = zeros(n_fpaths, height, width);
                inputs_agg = zeros(n_fpaths, height, width);
                diffs_agg = zeros(n_fpaths, height, width);
            end
            
            % add to the placeholders
            recs_agg(frame_num, :, :) = rec;
            inputs_agg(frame_num, :, :) = inputs;
            diffs_agg(frame_num, :, :) = diff;
            
        end  % for frame_num = 1:n_fpaths
            
        % scale each sample
        recs_agg = (recs_agg - min(min(min(recs_agg)))) / (max(max(max(recs_agg))) - min(min(min(recs_agg))));
        inputs_agg = (inputs_agg - min(min(min(inputs_agg)))) / (max(max(max(inputs_agg))) - min(min(min(inputs_agg))));
        diffs_agg = (diffs_agg - min(min(min(diffs_agg)))) / (max(max(max(diffs_agg))) - min(min(min(diffs_agg))));

        % save each sample
        for frame_num = 1:n_fpaths
            if frame_num == 1
                imwrite(
                    squeeze(inputs_agg(frame_num, :, :)), 
                    input_fpath_save, 
                    'gif', 
                    'writemode', 
                    'overwrite', 
                    'Loopcount', 
                    inf, 
                    'DelayTime', 
                    0.5
                );
                imwrite(
                    squeeze(recs_agg(frame_num, :, :)), 
                    rec_fpath_save, 
                    'gif', 
                    'writemode', 
                    'overwrite', 
                    'Loopcount', 
                    inf, 
                    'DelayTime', 
                    0.5
                );
                imwrite(
                    squeeze(diffs_agg(frame_num, :, :)), 
                    diff_fpath_save, 
                    'gif', 
                    'writemode', 
                    'overwrite', 
                    'Loopcount', 
                    inf, 
                    'DelayTime', 
                    0.5
                );

            else
                imwrite(
                    squeeze(inputs_agg(frame_num, :, :)), 
                    input_fpath_save, 
                    'gif', 
                    'writemode', 
                    'append', 
                    'DelayTime', 
                    0.1
                );
                imwrite(
                    squeeze(recs_agg(frame_num, :, :)), 
                    rec_fpath_save, 
                    'gif', 
                    'writemode', 
                    'append', 
                    'DelayTime', 
                    0.1
                );
                imwrite(
                    squeeze(diffs_agg(frame_num, :, :)), 
                    diff_fpath_save, 
                    'gif', 
                    'writemode', 
                    'append', 
                    'DelayTime', 
                    0.1
                );
            end

        end  % for frame_num = 1:n_fpaths

    end  % for batch_num = 1:n_inputs
