-- Main script that loads other scripts, sets some of the options automatically.
-- User should typically set the following options to start training/testing a model:
-- 'expName', 'dataset', 'split', 'stream'. See opts.lua for others.
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
paths.dofile('trainplot/TrainPlotter.lua')

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

local nChannels
if(opt.stream == 'flow')    then opt.mean =  0; nChannels = 2
elseif(opt.stream == 'rgb') then opt.mean = 96; nChannels = 3; opt.coeff = 255 end

opt.save            = paths.concat(opt.logRoot, opt.dataset, opt.expName)
opt.cache           = paths.concat(opt.logRoot, opt.dataset, 'cache', opt.stream)
opt.data            = paths.concat(opt.dataRoot, opt.dataset, 'splits', 'split' .. opt.split)
opt.framesRoot      = paths.concat(opt.dataRoot, opt.dataset, opt.stream, 't7')
opt.forceClasses    = torch.load(paths.concat(opt.dataRoot, opt.dataset, 'annot/forceClasses.t7'))
opt.loadSize        = {nChannels, opt.nFrames, opt.loadHeight,      opt.loadWidth}
opt.sampleSize      = {nChannels, opt.nFrames, opt.sampleHeight, opt.sampleWidth}

paths.dofile(opt.LRfile)

-- Testing final predictions
if(opt.evaluate) then
    opt.save         = paths.concat(opt.logRoot, opt.dataset, opt.expName, 'test_' .. opt.modelNo .. '_slide' .. opt.slide)
    opt.cache        = paths.concat(opt.logRoot, opt.dataset, 'cache', 'test', opt.stream)
    opt.scales       = false
    opt.crops10      = true
    opt.testDir      = 'test_' .. opt.loadSize[2] .. '_' .. opt.slide
    opt.retrain      = paths.concat(opt.logRoot, opt.dataset, opt.expName, 'model_' .. opt.modelNo .. '.t7')
    opt.finetune     = 'none'
end

-- Continue training (epochNumber has to be set for this option)
if(opt.continue) then
    print('Continuing from epoch ' .. opt.epochNumber)
    opt.retrain = opt.save .. '/model_' .. opt.epochNumber -1 ..'.t7'
    opt.finetune = 'none'
    opt.optimState = opt.save .. '/optimState_'.. opt.epochNumber -1  ..'.t7'
    local backupDir = opt.save .. '/delete' .. os.time()
    os.execute('mkdir -p ' .. backupDir)
    os.execute('cp ' .. opt.save .. '/train.log ' ..backupDir)
    os.execute('cp ' .. opt.save .. '/' .. opt.testDir..'.log ' ..backupDir)
    os.execute('cp ' .. opt.save .. '/plot.json ' ..backupDir)
end

os.execute('mkdir -p ' .. opt.save)
os.execute('mkdir -p ' .. opt.cache)
opt.plotter = TrainPlotter.new(paths.concat(opt.save, 'plot.json'))
opt.plotter:info({created_time=io.popen('date'):read(), tag=opt.expName})
print(opt)
print('Saving everything to: ' .. opt.save)
torch.save(paths.concat(opt.save, 'opt' .. os.time() .. '.t7'), opt)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)

paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('test.lua')

if(not opt.evaluate) then
    -- Training
    paths.dofile('train.lua')
    epoch = opt.epochNumber
    for i=1,opt.nEpochs do
        train()
        test()
        os.execute('scp ' .. paths.concat(opt.save, 'plot.json') .. ' ' .. paths.concat('trainplot/plot-data/', opt.dataset, opt.expName:gsub('%W','') ..'.json'))
        epoch = epoch + 1
    end
else
    -- Testing final predictions
    test()
end