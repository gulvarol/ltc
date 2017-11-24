require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

-- Create Network
-- If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    model = torch.load(opt.retrain)
else
    paths.dofile('models/' .. opt.netType .. '.lua')
    print('=> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel() -- for the model creation code, check the models/ folder
    if opt.backend == 'cudnn' then
        require 'cudnn'
        cudnn.convert(model, cudnn)
    elseif opt.backend == 'cunn' then
        require 'cunn'
        model = model:cuda()
    elseif opt.backend ~= 'nn' then
        error'Unsupported backend'
    end
end

-- Criterion
criterion = nn.ClassNLLCriterion()

-- Finetuning
if(opt.finetune == 'last' or opt.finetune == 'whole') then
    -- remove last two layers and add new ones (for new nClasses)
    -- either freeze previous layers (last) or train all layers (whole)
    local n_units = model:get(20):parameters()[2]:size()[1] -- 2048
    print('=> Model: Removing last two layers.')
    model:remove()
    model:remove()
   
    if(opt.finetune == 'last') then
        features = model:clone()
        features:evaluate()
        if(opt.lastlayer == 'none') then
            print('=> Model: Adding fc layer and logsoftmax to train only last layer.')
            model = nn.Sequential()
            model:add(nn.Linear(n_units, nClasses))
            model:add(nn.LogSoftMax()) 
        else
            print('=> Model: Adding pre-trained fc layer to train only last layer.')
            model = torch.load(opt.lastlayer)
        end
    elseif(opt.finetune == 'whole') then
        if(opt.lastlayer == 'none') then
            print('=> Model: Adding fc layer and logsoftmax to train whole network.')
            model:add(nn.Linear(n_units, nClasses))
            model:add(nn.LogSoftMax())
        else
            print('=> Model: Adding pre-trained fc layer to train whole network.')
            local classifier = torch.load(opt.lastlayer)
            model:add(classifier)
        end
    end
end

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

model = model:cuda()
criterion:cuda()

collectgarbage()
