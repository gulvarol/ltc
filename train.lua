require 'optim'

-- Setup a reused optimization state. If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay,
    alpha = 0.99,
    epsilon = 1e-8
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    for _, row in ipairs(opt.regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local acc, loss

-- train - this function handles the high-level training loop
function train()
	print('==> training epoch # ' .. epoch)
	local params, newRegime = paramsForEpoch(epoch)

	if newRegime then
		optimState = {
		learningRate = params.learningRate,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = params.weightDecay,
		alpha = 0.99,
		epsilon = 1e-8
		}
	end

	batchNumber = 0
	cutorch.synchronize()

	-- set the dropouts to training mode
	model:training()
	model:cuda()

	local tm = torch.Timer()
	acc = 0
	loss = 0
	for i=1,opt.epochSize do
		-- queue jobs to data-workers
		donkeys:addjob(
		-- the job callback (runs in data-worker thread)
		function()
			local inputs, labels = trainLoader:sample(opt.batchSize)
			return inputs, labels
		end,
		-- the end callback (runs in the main thread)
		trainBatch
		)
	end

	donkeys:synchronize()
	cutorch.synchronize()

	acc = acc * 100 / (opt.batchSize * opt.epochSize)
	loss = loss / opt.epochSize

	trainLogger:add{
		['epoch'] = epoch,
		['acc']   = acc,
		['loss']  = loss,
		['LR']    = optimState.learningRate
	}
	print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
						.. 'Loss: %.2f \t '
						.. 'Acc: %.2f\t\n',
						epoch, tm:time().real, loss, acc))

	opt.plotter:add('accuracy', 'train', epoch, acc)
	opt.plotter:add('loss', 'train', epoch, loss)
	opt.plotter:add('LR', 'train', epoch, optimState.learningRate)

	-- save model
	collectgarbage()
	model:clearState()
	torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
	torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
	cutorch.synchronize()
	collectgarbage()
	local dataLoadingTime = dataTimer:time().real
	timer:reset()

	-- transfer over to GPU
	inputs:resize(inputsCPU:size()):copy(inputsCPU)
	labels:resize(labelsCPU:size()):copy(labelsCPU)

	-- Compute the output of the fixed layers as input to the last layer model
	if(opt.finetune == 'last') then
		inputs = features:forward(inputs)
	end

	local lossBatch, outputs
	feval = function(x)
		model:zeroGradParameters()
		outputs = model:forward(inputs)
		lossBatch = criterion:forward(outputs, labels)
		local gradOutputs = criterion:backward(outputs, labels)
		model:backward(inputs, gradOutputs)
		return lossBatch, gradParameters
	end
	optim[opt.optimMethod](feval, parameters, optimState)

	cutorch.synchronize()
	batchNumber = batchNumber + 1
	loss = loss + lossBatch
	local accBatch = 0
	do
		outputs = outputs:view(opt.batchSize, -1) -- useful for opt.batchSize == 1
		local _,scores_sorted = outputs:float():sort(2, true) -- descending
    	for i=1,opt.batchSize do
			if scores_sorted[i][1] == labelsCPU[i] then -- correct prediction
				acc = acc + 1; 
				accBatch = accBatch + 1
			end
		end
		accBatch = accBatch * 100 / opt.batchSize;
	end
	print(('Epoch: Training [%d][%d/%d]\tTime %.3f Loss %.4f Acc: %.2f LR %.0e DataLoadingTime %.3f'):format(
			epoch, batchNumber, opt.epochSize, timer:time().real, lossBatch, accBatch,
			optimState.learningRate, dataLoadingTime))
	dataTimer:reset()
end