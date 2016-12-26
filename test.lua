testLogger = optim.Logger(paths.concat(opt.save, opt.testDir .. '.log'))

local batchNumber
local acc, loss
local timer = torch.Timer()

function test()
	local optimState 
	batchNumber = 0
	cutorch.synchronize()
	timer:reset()

	-- set the dropouts to evaluate mode
	model:evaluate()
	if(opt.crops10) then nDiv = 10 else nDiv = 1 end
	local N = nTest/torch.floor(opt.batchSize/nDiv) -- nTest is set in data.lua

	if(opt.evaluate) then
		print('==> testing final predictions')
		clipScores = torch.Tensor(N, nClasses)
	else
		optimState = torch.load(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'))
		print('==> validation epoch # ' .. epoch)
	end

	acc  = 0
	loss = 0
	for i=1, N do
		local indexStart = (i-1) * torch.floor(opt.batchSize/nDiv) + 1
		local indexEnd = (indexStart + torch.floor(opt.batchSize/nDiv) - 1)
		donkeys:addjob(
			-- work to be done by donkey thread
			function()
				local inputs, labels, indices = testLoader:get(indexStart, indexEnd)
				return inputs, labels, indices
			end,
		-- callback that is run in the main thread once the work is done
		testBatch
		)
	end

	donkeys:synchronize()
	cutorch.synchronize()

	acc  = acc * 100 / nTest
	loss = loss / (nTest/torch.floor(opt.batchSize/nDiv)) -- because loss is calculated per batch

	if(not opt.evaluate) then
		testLogger:add{
			['epoch'] = epoch,
			['acc'] = acc,
			['loss'] = loss,
			['LR'] = optimState.learningRate
		}
		opt.plotter:add('accuracy', 'test', epoch, acc)
		opt.plotter:add('loss', 'test', epoch, loss)
		print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t Loss: %.2f \t Acc: %.2f\n',
			epoch, timer:time().real, loss, acc))
	else
		paths.dofile('donkey.lua')
		local videoAcc = testLoader:computeAccuracy(clipScores)
		local result = {}
		result.accuracy = videoAcc
		result.scores = clipScores
		torch.save(paths.concat(opt.save, 'result.t7'), result)
		testLogger:add{
			['clipAcc'] = acc,
			['videoAcc'] = videoAcc
		}
		print(string.format('[TESTING SUMMARY] Total Time(s): %.2f \t Loss: %.2f \t Clip Acc: %.2f \t Video Acc: %.2f\n',
			timer:time().real, loss, acc, videoAcc))
	end
end -- of test()
-----------------------------------------------------------------------------

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU, indicesCPU)
	if(opt.crops10) then
		batchNumber = batchNumber + torch.floor(opt.batchSize/10)
	else
		batchNumber = batchNumber + torch.floor(opt.batchSize)
	end
	inputs:resize(inputsCPU:size()):copy(inputsCPU)

	local outputs
	if(opt.finetune == 'last') then
		outputs = model:forward(features:forward(inputs))
	else
		outputs = model:forward(inputs)
	end

	if(opt.crops10) then
	    outputs 	= torch.reshape(outputs, outputs:size(1)/10, 10, outputs:size(2))
	    outputs 	= torch.mean(outputs, 2):view(opt.batchSize/10, -1) -- mean over 10 crops
	    labelsCPU 	= labelsCPU:index(1, torch.range(1,  labelsCPU:size(1), 10):long())
	    indicesCPU 	= indicesCPU:index(1, torch.range(1, indicesCPU:size(1), 10):long())
	else
		outputs = outputs:view(opt.batchSize, -1) -- useful for opt.batchSize == 1
	end
	
	labels:resize(labelsCPU:size()):copy(labelsCPU)
	local lossBatch = criterion:forward(outputs, labels) 
	cutorch.synchronize()
   	loss = loss + lossBatch

   	local scoresCPU = outputs:float() -- N x 101
  	local gt, pred

	local _, scores_sorted = scoresCPU:sort(2, true)
	for i=1,scoresCPU:size(1) do
		gt = labelsCPU[i]                    -- ground truth class
		pred = scores_sorted[i][1]           -- predicted class
		if pred == gt then acc = acc + 1 end -- correct prediction

		if(opt.evaluate) then
		    clipScores[indicesCPU[i]] = scoresCPU[i]
		end
	end

	if(opt.evaluate) then
		print(string.format('Testing [%d/%d] \t Loss %.4f \t Acc %.2f', batchNumber, nTest, lossBatch, 100*acc/batchNumber))
	else
		print(string.format('Epoch: Testing [%d][%d/%d] \t Loss %.4f \t Acc %.2f', epoch, batchNumber, nTest, lossBatch, 100*acc/batchNumber))
	end
	collectgarbage()
end