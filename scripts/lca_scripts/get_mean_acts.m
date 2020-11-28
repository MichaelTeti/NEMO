function [mean_acts, sorted_acts, sorted_inds] = get_mean_acts(openpv_fpath, act_fpath, save_dir = '',
                                                 sparsity = true, shared = true, display = true)
    
    %{
        Reads in the .pvp file with the activations in the checkpoint dir and computes
        the mean activation over the batch and over space of each neuron.
        
        Args:
            openpv_fpath: The path to the OpenPV/mlab/util directory.
            act_fpath: The path to the <model_layer_name>_A.pvp file with each neuron's activation.
            save_dir: Where to save the figures if display is true.
            sparsity: If true, then will calculate mean sparsity for each neuron as well.
            shared: If true, will operate based on shared weights in the model.
            display: If true, then will display plots of the activations and sparsity.
    %}
    
    % add this to use OpenPV's matlab utilities for reading/writing .pvp files
    addpath(openpv_fpath);

    if ~exist(save_dir, 'dir')
        mkdir(dir = save_dir);
    end

    if ~strcmp(save_dir, '')
        if save_dir(end) ~= "/";
            save_dir = strcat(save_dir, "/");
        end
    end

    activity = readpvpfile(act_fpath);
    n_batch = size(activity, 1);
    n_feats = size(activity{1, 1}.values, 3);

    if shared
        mean_acts = zeros(n_feats, 1);
        
        if sparsity
            mean_sparsity = zeros(n_feats, 1);
        end

        for i_batch = 1:n_batch
            acts_i = activity{i_batch, 1}.values;
            mean_acts += squeeze(mean(mean(acts_i, 1), 2));
            
            if sparsity
                mean_sparsity += squeeze(mean(mean(acts_i ~= 0, 1), 2));
            end
            
        end

        mean_acts /= n_batch;
        [sorted_acts, sorted_inds] = sort(mean_acts, mode = 'descend');

        if display
            plot(sorted_acts, 'LineWidth', 8)
            xlabel("Sorted Feature Index")
            ylabel("Mean Activation Over Batch")
            print -dpng -color mean_acts.png
        end
        
        
        if sparsity
            mean_sparsity /= n_batch;
            [sorted_sparsity, sorted_inds] = sort(mean_sparsity, mode = 'descend');
            
            if display
                plot(sorted_sparsity, 'LineWidth', 8)
                xlabel("Sorted Feature Index")
                ylabel("Mean Sparsity Over Batch")
                print -dpng -color mean_sparsity.png
            end
            
        end
        

    else
        nx = size(activity{1, 1}.values, 1);
        ny = size(activity{1, 1}.values, 2);

        for feat = 1:n_feats
            mean_acts = zeros(nx * ny, 1);
            
            if sparsity 
                mean_sparsity = zeros(nx * ny, 1);
            end 

            for batch = 1:n_batch
                acts_batch_feat = activity{batch, 1}.values(:, :, feat);
                mean_acts += reshape(acts_batch_feat, nx * ny, 1);
                
                if sparsity
                    mean_sparsity += reshape(acts_batch_feat ~= 0, nx * ny, 1);
                end 
                
            end

            mean_acts /= n_batch;

            if display
                plot(mean_acts);
                xlabel("Spatial Index");
                ylabel("Mean Activation Over Batch");
                save_fpath = strcat(save_dir, "feature", num2str(feat), "_act.png");
                print(gcf, save_fpath, '-dpng');
            end
            
            if sparsity 
                mean_sparsity /= n_batch;
                
                if display
                    plot(mean_sparsity);
                    xlabel("Spatial Index");
                    ylabel("Mean Sparsity Over Batch");
                    save_fpath = strcat(save_dir, "feature", num2str(feat), "_sparsity.png");
                    print(gcf, save_fpath, '-dpng');
                end
                
            end 

        end

    end

end
