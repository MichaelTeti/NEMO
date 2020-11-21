function viz_shared_weights(openpv_path, checkpoint_path, save_path, key = '', 
    sorted = false, act_path = '')
    % Script to display PetaVision features and save them as an image or .gif.
    
    addpath(openpv_path)

    % check if the checkpoint path given exists
    if ~exist(checkpoint_path, 'dir')
       printf('Directory given to VizSharedWeights does not exist.')
       return
    end

    % add forward-slash to ckpt dir if not there
    if checkpoint_path(end) ~= '/'
      checkpoint_path = strcat(checkpoint_path, '/');
    end

    % if weight file key not given, just look for files that end with _W.pvp
    if strcmp(key, '')
        key = '*_W.pvp';
    end

    if sorted
        [~, ~, act_inds_sorted] = get_mean_acts(act_path)
    end

    % find the file paths in the checkpoint dir by the key
    fpaths = dir(strcat(checkpoint_path, key));
    n_fpaths = numel(fpaths);

    % loop through weight files
    for i_fpath = 1:n_fpaths
        % read in the file and load in weights
        fpath = strcat(checkpoint_path, fpaths(i_fpath, 1).name);
        w = readpvpfile(fpath);
        w = w{1, 1}.values{1, 1};

        % if sorting is desired, sort features by mean activation descending
        if sorted
            w = w(:, :, :, act_inds_sorted)
        end

        % initialize a grid to place the features in
        nxp = size(w, 1); nyp = size(w, 2); nfp = size(w, 3); nf = size(w, 4);
        grid_dim = ceil(sqrt(nf));
        grid_h = grid_dim * nyp; grid_w = grid_dim * nxp;
        grid = zeros(grid_h, grid_w, nfp);

        for i = 1:grid_dim
            for j = 1:grid_dim
                if (i-1)*grid_dim+j <= nf

                    % get the patch from the weight tensor
                    patch = w(:, :, :, (i-1)*grid_dim+j);

                    % scale each patch to [0, 1]
                    if ndims(patch) == 2
                        patch = transpose(patch);
                        patch = patch - min(min(patch));
                        patch = patch / (max(max(patch)) + 1e-6);
                    elseif ndims(patch) == 3 & size(patch, 3) == 3
                        patch = permute(patch, [2, 1, 3]);
                        patch = patch - min(min(min(patch)));
                        patch = patch / (max(max(max(patch))) + 1e-6);
                    end

                    % add patch to the grid
                    grid((i-1)*nyp+1:(i-1)*nyp+nyp, (j-1)*nxp+1:(j-1)*nxp+nxp, :) = patch;
                
                end  % if (i-1)
            end  % for j = 1:grid_dim
        end  % for i = 1:grid_dim

        if n_fpaths == 1
            imwrite(grid, save_path)
        else
            if i_fpath == 1
                imwrite(grid, save_path, 'gif', 'writemode', 'overwrite', 'Loopcount', inf, 'DelayTime', 0.5);
            else
                imwrite(grid, save_path, 'gif', 'writemode', 'append', 'DelayTime', 0.1);
            end
        end

    end
