function plot_num_active(openpv_fpath, fpath, save_dir = '')
    %{
        Reads in the .pvp file output by the model layer and computes 
        the number of neurons active over the display periods.
        
        Args:
            openpv_fpath: Path to the OpenPV/mlab/util directory.
            fpath: The path to the <model_layer_name>.pvp file.
            save_dir: Where to save the plot. If not given, will save 
                in current directory.
    %}
    
    % add this to use the openpv matlab utilities for reading .pvp files
    addpath(openpv_fpath);
    
    % if save_dir doesn't exist, then make it 
    if ~exist(save_dir, 'dir')
        mkdir(dir = save_dir);
    end

    % if save_dir doesn't end in forward-slash, add it 
    if ~strcmp(save_dir, '')
        if save_dir(end) ~= "/";
            save_dir = strcat(save_dir, "/");
        end
    end

    % check if the file exists
    if ~exist(fpath, 'file')
        error('fpath does not exist')
    end 

    % read in the pvpfile with the active neurons 
    act = readpvpfile(fpath);
    
    % loop through and aggregate the num active  
    n = size(act, 1);
    means = []; ses = [];
    
    for i = 1:n
        act_i = act{i, 1};
        
        if i > 1
            if times(end) ~= act_i.time
                times = [times; act_i.time];
                means = [means; mean(n_active_time)];
                ses = [ses; std(n_active_time) ./ sqrt(numel(n_active_time))];
                n_active_time = [size(act_i.values, 1)];
            else
                n_active_time = [n_active_time; size(act_i.values, 1)];
            end
        else
            times = [act_i.time];
            n_active_time = [size(act_i.values, 1)];
        end  

    end 


    if numel(times) ~= numel(means)
        times = times(1:end-1, :);
    end 

    
    % divide times by minimum time to show in terms of display period
    times = times ./ min(times);    

    % plot the figure
    figure (1)
    clf ()
    errorbar(times, means, ses);
    xlabel('Display Period Number');
    ylabel('Mean Number Active / Batch +/- SE');
    save_fpath = strcat(save_dir, num_active.png);
    print(gcf, save_fpath, '-dpng');
