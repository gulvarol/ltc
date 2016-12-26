local M = { }

function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Training script')
	cmd:text()
	cmd:text('Options:')
	------------ Path options -----------------------------------------------------------------------------------------------
	cmd:option('-dataRoot',        './datasets',			'Directory of datasets')
	cmd:option('-dataset',     	   'UCF101',            	'Name of the dataset (UCF101 | HMDB51)')
	cmd:option('-split', 		   1, 						'Split no (1 | 2 | 3)')
	cmd:option('-logRoot',         './log',           		'Log')
	cmd:option('-expName',         'exp',    				'Path to experiment')
	cmd:option('-data',            './datasets/UCF101', 	'Path to dataset') -- redundant
	cmd:option('-framesRoot',    './datasets/UCF101/flow/t7','Path to .t7 files') -- redundant
	cmd:option('-save',            './log/exp/save',    	'Directory in which to log experiments') -- redundant
	cmd:option('-cache',           './log/exp/cache',   	'Directory in which to cache data info') -- redundant
	cmd:option('-testDir',         'test',              	'Directory name of the test data')
	------------ General options --------------------
	cmd:option('-manualSeed',      2,                   	'Manually set RNG seed')
	cmd:option('-GPU',             1,                   	'Default preferred GPU')
	cmd:option('-backend',         'cudnn',             	'cudnn | cunn | nn')
	cmd:option('-nDonkeys',        8,                   	'Number of data loading threads (0 for debugging)') 
	cmd:option('-evaluate', 	   false, 					'Testing final predictions given model')
	cmd:option('-continue', 	   false, 					'Continuing stopped training from where it left')
	------------- Data processing options ----------------------------------------------------------------------------------
	cmd:option('-loadSize',        {2, 100, 67, 89},    	'(#channels, #frames, height, width) of video files') -- redundant
	cmd:option('-sampleSize',      {2, 100, 58, 58},    	'(#channels, #frames, height, width) of sampled videos') -- redundant
	cmd:option('-nFrames',	 	   100, 					'loadSize[2], sampleSize[2]')
	cmd:option('-loadHeight', 	   67, 						'loadSize[3]')
	cmd:option('-loadWidth', 	   89, 						'loadSize[4]')
	cmd:option('-sampleHeight',    58, 						'sampleSize[3]')
	cmd:option('-sampleWidth', 	   58, 						'sampleSize[4]')
	cmd:option('-stream',          'flow',                	'Whether the input is optical flow clip or rgb (flow | rgb)')
	cmd:option('-mean',            0,                   	'Mean pixel value of the data, or the path to the mean file')
	cmd:option('-perframemean',    true,                	'Whether the mean per frame should be subtracted')
	cmd:option('-minmax',          true,                	'Minmax normalization')
	cmd:option('-padType',         'copy',              	'Padding type for clips < #frames. (zero | copy)')
	cmd:option('-coeff',           1,                   	'Scalar multiplication of the input')
	cmd:option('-scales',          {1.0, 0.875, 0.75, 0.66},'Multiscale cropping coefficients (false | {list})')
	cmd:option('-slide',           4,                   	'Sliding window stride at test time')
	cmd:option('-crops10',         false,                	'Whether the test will be done on 10 crops or the center crop.')
	cmd:option('-bgr', 			   false, 					'BGR order (e.g. for C3D model ported from caffe)')
	------------- Training options -----------------------------------------------------------------------------------------
	cmd:option('-nEpochs',         50,                  	'Number of total epochs to run')
	cmd:option('-epochSize',       9000,                	'Number of batches per epoch')
	cmd:option('-epochNumber',     1,                   	'Manual epoch number (useful on restarts)')
	cmd:option('-batchSize',       10,                  	'Mini-batch size (1 = pure stochastic)')
	---------- Optimization options ----------------------------------------------------------------------------------------
	cmd:option('-optimMethod',     'sgd', 				    'sgd | rmsprop ...')
	cmd:option('-LR',              0.0,                 	'Learning rate; if set, overrides default LR/WD recipe')
	cmd:option('-momentum',        0.9,                 	'Momentum')
	cmd:option('-weightDecay',     5e-3,                	'Weight decay')
	cmd:option('-dropout',         0.9,                 	'Dropout')
	cmd:option('-LRfile', 		   './LR/UCF101/flow_d9.lua','Path to file where opt.regimes is defined')
	cmd:option('-regimes',         {{  1,      10,   3e-3,   5e-3, }, -- redundant
	                                {  11,     20,   3e-4,   5e-3  },
	                                {  21,    1e8,   3e-5,   5e-4 },}, 'Learning rate and weight decay regimes')
	---------- Model options ----------------------------------------------------------------------------------------------
	cmd:option('-netType',        'ltc',                	'Options: models/ directory')
	cmd:option('-retrain',        'none',               	'Path to model to retrain/load')
	cmd:option('-finetune',       'none',               	'Finetuning mode (none | last | whole)')
	cmd:option('-lastlayer',      'none',               	'Path to pre-trained last layer | none')
	cmd:option('-optimState',     'none',               	'Path to an optimState to reload from')
	cmd:option('-modelNo', 		  1, 						'Epoch number for the model to test')
	cmd:text()
	--plotter

	local opt = cmd:parse(arg or {})
	-- add commandline specified options
	opt.save = paths.concat(opt.cache,
	                        cmd:string('alexnet12', opt,
	                                   {retrain=true, optimState=true, cache=true, data=true}))
	-- add date/time
	opt.save = paths.concat(opt.save, ',' .. os.date():gsub(' ',''))
	return opt
end

return M
