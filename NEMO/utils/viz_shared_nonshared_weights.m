function viz_shared_nonshared_weights(checkpoint_path, save_dir, nx, ny, key = '', sorted = false, act_path = '')
    % make sure key has asterisks (*) where appropriate if needed

    addpath('/home/mteti/OpenPV/mlab/util');

    if ~exist(checkpoint_path, 'dir')
       printf('Directory given to VizSharedWeights does not exist.')
       return
    end

    if checkpoint_path(end) ~= '/'
      checkpoint_path = strcat(checkpoint_path, '/');
    end
    
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end 
    
    if ~strcmp(save_dir(end), "/")
        save_dir = strcat(save_dir, "/");
    end

    if strcmp(key, '')
        key = '*_W.pvp';
    else
        key = strcat(key, '_W.pvp');
    end

    if sorted
        [~, ~, act_inds_sorted] = get_mean_acts(act_path)
    end

    fpaths = dir(strcat(checkpoint_path, key));
    n_fpaths = numel(fpaths);

    for i_fpath = 1:n_fpaths
        fpath = strcat(checkpoint_path, fpaths(i_fpath, 1).name);
        w = readpvpfile(fpath);
        w = w{1, 1}.values{1, 1};

        if sorted
            w = w(:, :, :, act_inds_sorted)
        end

        nxp = size(w, 1); nyp = size(w, 2); nfp = size(w, 3); nf = size(w, 4) / nx / ny;
        grid_x = nx * nxp; grid_y = ny * nyp;
        w = reshape(w, nxp, nyp, nfp, nf, nx, ny);
        w = permute(w, [2, 1, 3, 4, 6, 5])
        
        for f = 1:nf
            grid = zeros(grid_y, grid_x, nfp);

            for i = 1:ny
                for j = 1:nx
                    patch = w(:, :, :, f, i, j);

                    if ndims(patch) == 2
                        patch = patch - min(min(patch));
                        patch = patch / (max(max(patch)) + 1e-6);
                    elseif ndims(patch) == 3 & size(patch, 3) == 3
                        patch = patch - min(min(min(patch)));
                        patch = patch / (max(max(max(patch))) + 1e-6);
                    end

                    grid((i-1) * nyp + 1: i * nyp, (j-1) * nxp + 1 : j * nxp, :) = patch;
                    
                end  % for j = 1:grid_dim
            end  % for i = 1:grid_dim
            
            % linearly scale the values to the range [0, 1]
            % grid = (grid - min(min(min(grid)))) / (max(max(max(grid))) - min(min(min(grid))));
            
            % write out the grid for that feature
            if n_fpaths == 1
                imwrite(grid, strcat(save_dir, "feature_", num2str(f), ".png"))
            else
                if i_fpath == 1
                    imwrite(grid, strcat(save_dir, "feature_", num2str(f), ".gif"), 'gif', 'writemode', 'overwrite', 'Loopcount', inf, 'DelayTime', 0.5);
                else
                    imwrite(grid, strcat(save_dir, "feature_", num2str(f), ".gif"), 'gif', 'writemode', 'append', 'DelayTime', 0.1);
                end
            end
            
        end  % for f = 1:nf

    end  % for i_path in ...
