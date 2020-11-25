function viz_reconstructions(openpv_path, checkpoint_dir, save_dir, rec_key, input_layer_prefix)
    %{
        Reads in inputs and reconstructions from a checkpoint and writes them out.
         
        Args:
            openpv_path: The path to OpenPV/mlab/util.
            checkpoint_dir: The path to the checkpoint directory where the input and 
                recon .pvp files are. 
            save_dir: The directory where the inputs and recons will be saved in separate
                subdirectories.
            rec_key: The key used to find the recon .pvp files (e.g. Frame*Recon_A.pvp).
            input_layer_prefix: The name of the input layer minus the frame number. For 
                example, a model trained on 9 video frame inputs will have 9 input layers
                called Frame0...Frame8. This argument would then be "Frame". 
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

    % make different dirs to save inputs and recons
    rec_dir = strcat(save_dir, 'Recons/');
    input_dir = strcat(save_dir, 'Inputs/');

    if ~exist(rec_dir, 'dir')
        mkdir(rec_dir);
    end

    if ~exist(input_dir, 'dir')
        mkdir(input_dir);
    end

    % find the fpaths with recons and inputs in checkpoint dir
    rec_fpaths = dir(strcat(checkpoint_dir, rec_key));
    n_inputs = size(readpvpfile(strcat(checkpoint_dir, rec_fpaths(1, 1).name)), 1);
    n_fpaths = numel(rec_fpaths);

    for i_input = 1:n_inputs
        for i_fpath = 1:n_fpaths
            rec_fpath = strcat(checkpoint_dir, rec_fpaths(i_fpath, 1).name);
            input_fpath = strcat(checkpoint_dir, strcat(input_layer_prefix, int2str(i_fpath - 1), '_A.pvp'));

            rec = readpvpfile(rec_fpath);
            input = readpvpfile(input_fpath);

            rec = rec{i_input, 1}.values;
            input = input{i_input, 1}.values;

            rec = transpose(rec);
            input = transpose(input);
            
            rec = (rec - min(min(min(rec)))) / (max(max(max(rec))) - min(min(min(rec))));
            input = (input - min(min(min(input)))) / (max(max(max(input))) - min(min(min(input))));

            rec_fpath = strcat(rec_dir, strcat(int2str(i_input), '.gif'));
            input_fpath = strcat(input_dir, strcat(int2str(i_input), '.gif'));

            if i_fpath == 1
                imwrite(input, input_fpath, 'gif', 'writemode', 'overwrite', 'Loopcount', inf, 'DelayTime', 0.5);
                imwrite(rec, rec_fpath, 'gif', 'writemode', 'overwrite', 'Loopcount', inf, 'DelayTime', 0.5);

            else
                imwrite(input, input_fpath, 'gif', 'writemode', 'append', 'DelayTime', 0.1);
                imwrite(rec, rec_fpath, 'gif', 'writemode', 'append', 'DelayTime', 0.1);
            end

        end

    end
