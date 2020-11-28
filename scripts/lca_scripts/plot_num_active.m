function plot_num_active(fpath)
    
    addpath('/home/mteti/OpenPV/mlab/util');

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
    print -dpng num_active.png
