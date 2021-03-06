-- Path to OpenPV. Replace this with an absolute path.
package.path = "/home/mteti/OpenPV/parameterWrapper/?.lua";
local pv = require "PVModule";


-------------------------------------------------------------------------------
--------------------------- Model Parameters and Setup ------------------------
-------------------------------------------------------------------------------

-- Input Image Vars
local inputFeatures             = 1;
local inputHeight               = 32;
local inputPathPrefix           = "filenames_frame";
local inputPathExt              = ".txt";
local inputWidth                = 64;
local nbatch                    = 128;
local numImages                 = 762246;


-- Model Vars
local AMax                      = infinity;
local AMin                      = 0;
local dictionarySize            = 8*24*256;
local displayMultiple           = 1;
local displayPeriod             = 3000;
local errorLayerPhase           = 2;
local growthFactor              = 0.005;
local initFromCkpt              = false;
local initFromCkptPath          = nil;
local initFromFile              = true;
local initFromFilePath          = "NonsharedWeights";
local initFromFilePrefix        = "S1";
local inputLayerPhase           = 1;
local learningRate              = 0.01;
local modelType                 = "LCA";
local modelLayerPhase           = 3;
local momentumTau               = 100;
local numEpochs                 = 10;
local patchSizeX                = 64;
local patchSizeY                = 32;
local plasticity                = true;
local reconLayerPhase           = 4;
local sharedWeights             = true;
local startFrame                = 1380;
local startTime                 = 0;
local stopTime                  = math.ceil(numImages / nbatch) * displayPeriod *
                                    displayMultiple * numEpochs;
local strideX                   = 64;
local strideY                   = 32;
local temporalPatchSize         = 9;
local threshType                = "soft";
local timeConstantTau           = 3000;
local useGPU                    = true;
local VThresh                   = 0.125;


--Probes and Checkpointing
local writeAdaptiveThreshProbe  = true;
local checkpointPeriod          = displayPeriod * displayMultiple;
local deleteOldCheckpoints      = true;
local writeEnergyProbe          = true;
local error2ModelWriteStep      = -1; --displayPeriod;
local errorWriteStep            = -1; --displayPeriod;
local writeFirmThreshProbe      = true;
local inputWriteStep            = -1; --displayPeriod;
local writeL2Probe              = true;
local model2ErrorWriteStep      = -1; --displayPeriod;
local model2ReconWriteStep      = displayPeriod;
local modelWriteStep            = displayPeriod;
local numCheckpointsKept        = 1;
local runNote                   = nil;
local runVersion                = 1;

-- where checkpoints for this run will be written
local outputPath                = "runs/run" .. runVersion .. "_" .. modelType;

-- option to add a note to the run's checkpoint dir
if runNote then
    outputPath = outputPath .. "_" .. runNote;
end

-- set VWidth based on desired threshold type
if threshType == "soft" then
    VWidth = infinity;
elseif threshType == "firm" then
    VWidth = VThresh;
end

-- add forward slash to path if not given
if initFromFile and string.sub(initFromFilePath, -1) ~= "/" then
    initFromFilePath = initFromFilePath .. "/";
end


-- create some names for the model and the input layer
if modelType == "LCA" then
    modelPrefix = "S";
elseif modelType == "STRF" then
    modelPrefix = "H";
end

local modelIndex = "1";
local modelLayer = modelPrefix .. modelIndex;
local inputPrefix = "Frame";
local inputLayer0 = inputPrefix .. "0";


-------------------------------------------------------------------------------
-------------------------- Initialize Hyper-Column ----------------------------
-------------------------------------------------------------------------------

local pvParams = {
    column = {
        groupType                           = "HyPerCol";
        startTime                           = startTime;
        dt                                  = 1;
        stopTime                            = stopTime;
        progressInterval                    = checkpointPeriod;
        writeProgressToErr                  = true;
        verifyWrites                        = false;
        outputPath                          = outputPath;
        printParamsFilename                 = "model.params";
        randomSeed                          = 10000000;
        nx                                  = inputWidth;
        ny                                  = inputHeight;
        nbatch                              = nbatch;
        initializeFromCheckpointDir         = nil;
        checkpointWrite                     = true;
        checkpointWriteDir                  = outputPath .. "/Checkpoints";
        checkpointWriteTriggerMode          = "step";
        checkpointWriteStepInterval         = checkpointPeriod;
        deleteOlderCheckpoints              = deleteOldCheckpoints;
        numCheckpointsKept                  = numCheckpointsKept;
        suppressNonplasticCheckpoints       = false;
        writeTimescales                     = true;
        errorOnNotANumber                   = true;
    }
}

if initFromCkpt then
   pvParams.column.initializeFromCheckpointDir = initFromCkptPath;
end

-------------------------------------------------------------------------------
---------------------------------- Probes -------------------------------------
-------------------------------------------------------------------------------

-- adaptive timescales probe
pv.addGroup(pvParams,
    "AdaptiveTimeScales", {
        groupType                           = "LogTimeScaleProbe";
        targetName                          = "EnergyProbe";
        message                             = NULL;
        textOutputFlag                      = false;
        probeOutputFile                     = "AdaptiveTimeScales.txt";
        triggerLayerName                    = inputLayer0;
        triggerOffset                       = 0;
        baseMax                             = 1.1;
        baseMin                             = 1.0;
        tauFactor                           = 0.025;
        growthFactor                        = growthFactor;
        logThresh                           = 10.0;
        logSlope                            = 0.01;
    }
)

if writeAdaptiveThreshProbe then
    pvParams["AdaptiveTimeScales"].textOutputFlag = true;
end

-- energy probe
pv.addGroup(pvParams,
    "EnergyProbe", {
        groupType                       = "ColumnEnergyProbe";
        message                         = nil;
        textOutputFlag                  = false;
        probeOutputFile                 = "EnergyProbe.txt";
        triggerLayerName                = nil;
        energyProbe                     = nil;
    }
)

if writeEnergyProbe then
    pvParams["EnergyProbe"].textOutputFlag = true;
end

-- firm threshold probe
pv.addGroup(pvParams,
    modelLayer .. "FirmThreshProbe", {
        groupType                       = "FirmThresholdCostFnLCAProbe";
        targetLayer                     = modelLayer;
        message                         = NULL;
        textOutputFlag                  = false;
        probeOutputFile                 = modelLayer .. "FirmThreshProbe.txt";
        triggerLayerName                = NULL;
        energyProbe                     = "EnergyProbe";
        maskLayer                       = NULL;
    }
)

if writeFirmThreshProbe then
    pvParams[modelLayer .. "FirmThreshProbe"].textOutputFlag = true;
end

-------------------------------------------------------------------------------
------------------------ Create Layers and Connections ------------------------
-------------------------------------------------------------------------------

-------------------------------- Model Layer ----------------------------------
pv.addGroup(pvParams,
    modelLayer, {
        groupType                           = "HyPerLCALayer";
        nxScale                             = 1 / strideX;
        nyScale                             = 1 / strideY;
        nf                                  = dictionarySize;
        phase                               = modelLayerPhase;
        mirrorBCflag                        = false;
        valueBC                             = 0;
        initializeFromCheckpointFlag        = false;
        InitVType                           = "ConstantV";
        valueV                              = 0.0 * VThresh;
        triggerLayerName                    = NULL;
        writeStep                           = modelWriteStep;
        initialWriteTime                    = modelWriteStep;
        sparseLayer                         = true;
        writeSparseValues                   = true;
        updateGpu                           = useGPU;
        dataType                            = nil;
        VThresh                             = VThresh;
        AMin                                = AMin;
        AMax                                = AMax;
        AShift                              = 0;
        VWidth                              = VWidth;
        clearGSynInterval                   = 0;
        timeConstantTau                     = timeConstantTau;
        selfInteract                        = true;
        adaptiveTimeScaleProbe              = nil;
    }
)

pvParams[modelLayer].adaptiveTimeScaleProbe  = "AdaptiveTimeScales";

if initFromCkpt then
    pvParams[modelLayer].initializeFromCheckpointFlag = true;
end


-------------------------- Recon, Error, and Input Layers ---------------------
for i_frame = 1, temporalPatchSize do

    -- Create layer names ... already defined modelLayer above
    inputLayer = inputPrefix .. i_frame - 1;
    reconLayer = inputLayer .. "Recon";
    errorLayer = reconLayer .. "Error";

    ----------------------------- Input Layer ---------------------------------
    local start_frame_index_array = {};
    for i_batch = 1,nbatch do
        start_frame_index_array[i_batch] = startFrame;
    end

    pv.addGroup(pvParams,
        inputLayer, {
          groupType                       = "ImageLayer";
    	    nxScale                         = 1;
    	    nyScale                         = 1;
    	    nf                              = inputFeatures;
    	    phase                           = inputLayerPhase;
    	    mirrorBCflag                    = true;
    	    writeStep                       = inputWriteStep;
    	    initialWriteTime                = inputWriteStep;
    	    sparseLayer                     = false;
    	    updateGpu                       = false;
    	    dataType                        = nil;
    	    inputPath                       = inputPathPrefix .. i_frame-1 .. inputPathExt;
    	    offsetAnchor                    = "cc";
    	    offsetX                         = 0;
    	    offsetY                         = 0;
    	    writeImages                     = 0;
    	    inverseFlag                     = false;
    	    normalizeLuminanceFlag          = true;
    	    normalizeStdDev                 = true;
    	    jitterFlag                      = 0;
    	    useInputBCflag                  = false;
    	    padValue                        = 0;
    	    autoResizeFlag                  = true;
    	    aspectRatioAdjustment           = "crop";
    	    interpolationMethod             = "bicubic";
    	    displayPeriod                   = displayPeriod * displayMultiple;
    	    batchMethod                     = "byList";
    	    start_frame_index               = start_frame_index_array;
    	    writeFrameToTimestamp           = true;
    	    resetToStartOnLoop              = false;
    	    initializeFromCheckpointFlag    = false;
        }
    )

    ---------------------------- Error Layer ----------------------------------
    pv.addGroup(pvParams,
        errorLayer, {
            groupType                        = "HyPerLayer";
            nxScale                          = 1;
            nyScale                          = 1;
            nf                               = inputFeatures;
            phase                            = errorLayerPhase;
            mirrorBCflag                     = false;
            valueBC                          = 0;
            -- initializeFromCheckpointFlag  = false;
            InitVType                        = "ZeroV";
            triggerLayerName                 = NULL;
            writeStep                        = errorWriteStep;
            initialWriteTime                 = errorWriteStep;
            sparseLayer                      = false;
            updateGpu                        = false;
            dataType                         = nil;
        }
    )

    ---------------------- L2Probe for the Error Layer ------------------------
    -- easier to just put this here instead of with the other probes
    pv.addGroup(pvParams,
        errorLayer .. "L2Probe", {
            groupType                   = "L2NormProbe";
            targetLayer                 = errorLayer;
            message                     = nil;
            textOutputFlag              = false;
            probeOutputFile             = errorLayer .. "L2Probe.txt";
            energyProbe                 = "EnergyProbe";
            coefficient                 = 0.5;
            maskLayerName               = nil;
            exponent                    = 2;
        }
    )

    if writeL2Probe then
        pvParams[errorLayer .. "L2Probe"].textOutputFlag = true;
    end

    -------------------------- Recon layer ------------------------------------
    pv.addGroup(pvParams,
        reconLayer, pvParams[errorLayer], {
            phase = reconLayerPhase;
        }
    )

    ------ Input Layer to Error Layer Excitatory Identity Connection ----------
    pv.addGroup(pvParams,
        inputLayer .. "To" .. errorLayer, {
            groupType                        = "IdentConn";
            preLayerName                     = inputLayer;
            postLayerName                    = errorLayer;
            channelCode                      = 0;
            scale                            = weightInit;
            delay                            = {0.000000};
        }
    )

    -------- Recon Layer to Error Layer Inhibitory Identity Connection --------
    pv.addGroup(pvParams,
        reconLayer .. "To" .. errorLayer, {
            groupType                        = "IdentConn";
            preLayerName                     = reconLayer;
            postLayerName                    = errorLayer;
            channelCode                      = 1;
            delay                            = {0.000000};
        }
    )

    ------- Error Layer to Model Layer Excitatory Transpose Connection --------
    pv.addGroup(pvParams,
        errorLayer .. "To" .. modelLayer, {
            groupType                        = "TransposeConn";
            preLayerName                     = errorLayer;
            postLayerName                    = modelLayer;
            channelCode                      = 0;
            delay                            = {0.000000};
            convertRateToSpikeCount          = false;
            receiveGpu                       = useGPU;
            updateGSynFromPostPerspective    = true;
            pvpatchAccumulateType            = "convolve";
            writeStep                        = error2ModelWriteStep;
            writeCompressedCheckpoints       = false;
            selfFlag                         = false;
            gpuGroupIdx                      = -1;
            originalConnName                 = modelLayer .. "To" .. errorLayer;
        }
    )

    ------------ Model Layer to Recon Layer Excitatory Connection -------------
    pv.addGroup(pvParams,
        modelLayer .. "To" .. reconLayer, {
            groupType                       = "CloneConn";
            preLayerName                    = modelLayer;
            postLayerName                   = reconLayer;
            channelCode                     = 0;
            writeStep                       = model2ReconWriteStep;
            delay                           = {0.000000};
            convertRateToSpikeCount         = false;
            receiveGpu                      = false;
            updateGSynFromPostPerspective   = false;
            pvpatchAccumulateType           = "convolve";
            writeCompressedCheckpoints      = false;
            selfFlag                        = false;
            originalConnName                = modelLayer .. "To" .. errorLayer;
        }
    )

    ----------- Model Layer to Error Layer Passive Update Connection ----------
    pv.addGroup(pvParams,
        modelLayer .. "To" .. errorLayer, {
            groupType                       = "MomentumConn";
            preLayerName                    = modelLayer;
            postLayerName                   = errorLayer;
            channelCode                     = -1;
            delay                           = {0.000000};
            numAxonalArbors                 = 1;
            convertRateToSpikeCount         = false;
            receiveGpu                      = false;
            sharedWeights                   = sharedWeights;
            initializeFromCheckpointFlag    = false;
            triggerLayerName                = inputLayer0;
            triggerOffset                   = 1;
            updateGSynFromPostPerspective   = false;
            pvpatchAccumulateType           = "convolve";
            writeStep                       = model2ErrorWriteStep;
            initialWriteTime                = model2ErrorWriteStep;
            writeCompressedCheckpoints      = false;
            selfFlag                        = false;
            shrinkPatches                   = false;
            normalizeMethod                 = "normalizeL2";
            strength                        = 1;
            normalizeArborsIndividually     = false;
            normalizeOnInitialize           = true;
            normalizeOnWeightUpdate         = true;
            rMinX                           = 0;
            rMinY                           = 0;
            nonnegativeConstraintFlag       = false;
            normalize_cutoff                = 0;
            normalizeFromPostPerspective    = false;
            minL2NormTolerated              = 0;
            keepKernelsSynchronized         = true;
            useMask                         = false;
            normalizeDw                     = true;
            timeConstantTau                 = momentumTau;
            momentumMethod                  = "viscosity";
            momentumDecay                   = 0;
            nxp                             = patchSizeX;
            nyp                             = patchSizeY;
            plasticityFlag                  = plasticity;
            weightInitType                  = "UniformRandomWeight";
            initWeightsFile                 = nil;
            wMinInit                        = -1.0;
            wMaxInit                        = 1.0;
            sparseFraction                  = 0.9;
            dWMax                           = learningRate;
        }
    )

    if initFromCkpt then
        pvParams[modelLayer .. "To" .. errorLayer].initializeFromCheckpointFlag = true;
    end

    if initFromFile then
        if initFromFilePrefix then
            filePath = initFromFilePath .. initFromFilePrefix .. "To" .. inputLayer .. "ReconError_W.pvp";
        else
            filePath = initFromFilePath .. modelLayer .. "To" .. inputLayer .. "ReconError_W.pvp";
        end

        pvParams[modelLayer .. "To" .. errorLayer].weightInitType = "FileWeight";
        pvParams[modelLayer .. "To" .. errorLayer].initWeightsFile = filePath;

    end

    if i_frame > 1 then
        pvParams[modelLayer .. "To" .. errorLayer].normalizeMethod = "normalizeGroup";
        pvParams[modelLayer .. "To" .. errorLayer].normalizeGroupName = modelLayer .. "To" .. inputLayer0 .. "Recon" .. "Error";

        if not initFromCkpt and not initFromFile then
            pvParams[modelLayer .. "To" .. errorLayer].wMinInit = 0;
            pvParams[modelLayer .. "To" .. errorLayer].wMaxInit = 0;
        end
    end


end -- i_frame


-- print out to the .param file
pv.printConsole(pvParams)
