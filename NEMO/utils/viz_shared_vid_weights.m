function viz_shared_vid_weights(checkpoint_path, save_path, key = '',
    clip_frame_0 = false, sorted_feats = false, act_path = '')
    %{ 
        Script to display PetaVision features and save them as an image or .gif.
    
        Args:
            checkpoint_path: The path to the checkpoint with the features to visualize.
            save_path: The file path to the saved .gif (e.g. "features.gif").
            key: A key to differentiate the weight files you want to visualize.
            clip_frame_0: If true, will clip frame zero's values to the min/max of the 
                subsequent frames. This is mainly used in the early parts of training 
                because every frame after zero is initialized with zeros. 
            sorted_feats: If true, will sort the features in the grid (descending) by mean 
                activation.
            act_path: Path to the model's feature maps in the checkpoint that will
                be used to sort the features in descending order if sorted is true.
    %}

    % check if the checkpoint path given exists
    if ~exist(checkpoint_path, 'dir')
       error('Directory given to VizSharedWeights does not exist.');
    end

    % add forward-slash to ckpt dir if not there
    if checkpoint_path(end) ~= '/'
      checkpoint_path = strcat(checkpoint_path, '/');
    end

    % if weight file key not given, just look for files that end with _W.pvp
    if strcmp(key, '')
        key = '*_W.pvp';
    end

    % if you want the features sorted by mean activation, we need to
    % read in the activity file and get the sorted indices based on activation
    if sorted_feats

        % check if the activity file exists
        if ~exist(act_path)
            error('act_path does not exist.');
        end

        [~, ~, act_inds_sorted] = get_mean_acts(act_path)
    end

    % find the file paths in the checkpoint dir by the key
    fpaths = dir(strcat(checkpoint_path, key));
    n_fpaths = numel(fpaths);

    % loop through weight files
    for fpath_num = 1:n_fpaths
        % read in the file and load in weights
        fpath = strcat(checkpoint_path, fpaths(fpath_num, 1).name);
        w = readpvpfile(fpath);
        w = w{1, 1}.values{1, 1};

        % switch height and width dim for image format
        w = permute(w, [2, 1, 3, 4]);

        % if sorting is desired, sort features by mean activation descending
        if sorted_feats
            w = w(:, :, :, act_inds_sorted)
        end

        % initialize a grid to place the features in if first file read in
        if fpath_num == 1
            nyp = size(w, 1); nxp = size(w, 2); nfp = size(w, 3); nf = size(w, 4);
            grid_dim = ceil(sqrt(nf));
            grid_h = grid_dim * nyp; grid_w = grid_dim * nxp;
            grid = zeros(n_fpaths, grid_h, grid_w, nfp);
        end

        % below we add each patch to the grid
        for i = 1:grid_dim
            for j = 1:grid_dim
                if (i-1)*grid_dim+j <= nf
                    % get the patch from the weight tensor and add to the grid
                    patch = w(:, :, :, (i-1)*grid_dim+j);
                    grid(fpath_num, (i-1)*nyp+1:(i-1)*nyp+nyp, (j-1)*nxp+1:(j-1)*nxp+nxp, :) = patch;
                end  
            end  
        end  
    
    end  % for fpath_num ...


    % clip frame 0 features if desired and scale feature values
    if clip_frame_0
        max_val = max(max(max(max(grid(2:end, :, :, :)))));
        min_val = min(min(min(min(grid(2:end, :, :, :)))));
        grid(grid > max_val) = max_val;
        grid(grid < min_val) = min_val;
    else
        max_val = max(max(max(max(grid))));
        min_val = min(min(min(min(grid))));
    end    

    grid = (grid - min_val) / (max_val - min_val);

    % write the .gif out
    for fpath_num = 1:n_fpaths
        if fpath_num == 1
            imwrite(
                squeeze(grid(fpath_num, :, :, :)),
                save_path,
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
                squeeze(grid(fpath_num, :, :, :)),
                save_path,
                'gif',
                'writemode',
                'append',
                'DelayTime',
                0.1
            );
        end  
        
    end  % for fpath_num = 1:n_fpaths
