function viz_reconstructions(openpv_path, checkpoint_dir, save_dir, rec_key)
    %{
        Reads in inputs and reconstructions from a checkpoint and writes them out.
         
        Args:
            openpv_path: The path to OpenPV/mlab/util.
            checkpoint_dir: The path to the checkpoint directory where the input and 
                recon .pvp files are. 
            save_dir: The directory where the inputs and recons will be saved in separate
                subdirectories.
            rec_key: The key used to find the recon .pvp files (e.g. Frame*Recon_A.pvp). 
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
    n_inputs = size(readpvpfile(strcat(checkpoint_dir, rec_fpaths(1, 1).name)), 1);
    n_fpaths = numel(rec_fpaths);

    % get the input layer name 
    input_layer_name = char(strsplit(rec_key, "*")(1, 1));

    for i_input = 1:n_inputs
        rec_fpath = strcat(rec_dir, strcat(int2str(i_input), '.gif'));
        input_fpath = strcat(input_dir, strcat(int2str(i_input), '.gif'));
        diff_fpath = strcat(diff_dir, strcat(int2str(i_input), '.gif'));
    
        for i_fpath = 1:n_fpaths
            % read in the inputs and recon for this batch sample and video frame
            rec_fpath = strcat(checkpoint_dir, rec_fpaths(i_fpath, 1).name);
            input_fpath = strcat(checkpoint_dir, strcat(input_layer_name, int2str(i_fpath - 1), '_A.pvp'));
            rec = readpvpfile(rec_fpath);
            inputs = readpvpfile(input_fpath);
            rec = rec{i_input, 1}.values;
            inputs = inputs{i_input, 1}.values;

            % go from x, y to y, x to save as images
            rec = transpose(rec);
            inputs = transpose(inputs);
            
            % get the difference
            diff = inputs - rec;
            
            % make a placeholder to aggregate frames from each batch sample
            % so we can scale them by statistics of all frames for that sample
            if i_fpath == 1
                height = size(rec, 1);
                width = size(rec, 2);
                recs_agg = zeros(n_fpaths, height, width);
                inputs_agg = zeros(n_fpaths, height, width);
                diffs_agg = zeros(n_fpaths, height, width);
            end
            
            % add to the placeholders
            recs_agg(i_fpath, :, :) = rec;
            inputs_agg(i_fpath, :, :) = inputs;
            diffs_agg(i_fpath, :, :) = diff;
            
        end  % for i_fpath = 1:n_fpaths
            
        % scale each sample
        recs_agg = (recs_agg - min(min(min(recs_agg)))) / (max(max(max(recs_agg))) - min(min(min(recs_agg))));
        inputs_agg = (inputs_agg - min(min(min(inputs_agg)))) / (max(max(max(inputs_agg))) - min(min(min(inputs_agg))));
        diffs_agg = (diffs_agg - min(min(min(diffs_agg)))) / (max(max(max(diffs_agg))) - min(min(min(diffs_agg))));

        % save each sample
        for i_fpath = 1:n_fpaths
            if i_fpath == 1
                imwrite(
                    squeeze(inputs_agg(i_fpath, :, :)), 
                    input_fpath, 
                    'gif', 
                    'writemode', 
                    'overwrite', 
                    'Loopcount', 
                    inf, 
                    'DelayTime', 
                    0.5
                );
                imwrite(
                    squeeze(recs_agg(i_fpath, :, :)), 
                    rec_fpath, 
                    'gif', 
                    'writemode', 
                    'overwrite', 
                    'Loopcount', 
                    inf, 
                    'DelayTime', 
                    0.5
                );
                imwrite(
                    squeeze(diffs_agg(i_fpath, :, :)), 
                    diff_fpath, 
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
                    squeeze(inputs_agg(i_fpath, :, :)), 
                    input_fpath, 
                    'gif', 
                    'writemode', 
                    'append', 
                    'DelayTime', 
                    0.1
                );
                imwrite(
                    squeeze(recs_agg(i_fpath, :, :)), 
                    rec_fpath, 
                    'gif', 
                    'writemode', 
                    'append', 
                    'DelayTime', 
                    0.1
                );
                imwrite(
                    squeeze(diffs_agg(i_fpath, :, :)), 
                    diff_fpath, 
                    'gif', 
                    'writemode', 
                    'append', 
                    'DelayTime', 
                    0.1
                );
            end

        end

    end  % for i_input = 1:n_inputs
