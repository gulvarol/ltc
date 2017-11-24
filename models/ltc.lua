function createModel()
    require 'cudnn'
    require 'cutorch'
    require 'cunn'
    require 'nn'

    local model = nn.Sequential()
    model:add(nn.VolumetricConvolution(opt.sampleSize[1], 64, 3, 3, 3, 1, 1, 1, 1, 1, 1)) --   3 x [W]    x [H]    x [T]   ->  64 x [W]    x   [H]  x [T]
    model:add(nn.Threshold(0, 0, true))
    model:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))                                  --  64 x [W]    x [H]    x [T]   ->  64 x [W/2]  x [H/2]  x [T]
    model:add(nn.VolumetricConvolution(64, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1))               --  64 x [W/2]  x [H/2]  x [T]   -> 128 x [W/2]  x [H/2]  x [T]
    model:add(nn.Threshold(0, 0, true))
    model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))                                  -- 128 x [W/2]  x [H/2]  x [T]   -> 128 x [W/4]  x [H/4]  x [T/2]
    model:add(nn.VolumetricConvolution(128, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1))              -- 128 x [W/4]  x [H/4]  x [T/2] -> 256 x [W/4]  x [H/4]  x [T/2]
    model:add(nn.Threshold(0, 0, true))
    model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))                                  -- 256 x [W/4]  x [H/4]  x [T/2] -> 256 x [W/8]  x [H/8]  x [T/4]
    model:add(nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1))              -- 256 x [W/8]  x [H/8]  x [T/4] -> 256 x [W/8]  x [H/8]  x [T/4]
    model:add(nn.Threshold(0, 0, true))
    model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))                                  -- 256 x [W/8]  x [H/8]  x [T/4] -> 256 x [W/16] x [H/16] x [T/8]
    model:add(nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1))              -- 256 x [W/16] x [H/16] x [T/8] -> 256 x [W/16] x [H/16] x [T/8]
    model:add(nn.Threshold(0, 0, true))
    model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))                                  -- 256 x [W/16] x [H/16] x [T/8] -> 256 x [W/32] x [H/32] x [T/16] 

    -- Size of the conv layers output
    local oT = math.floor(opt.sampleSize[2]/16); -- 4 times max pooling of 1/2
    local oH = math.floor(opt.sampleSize[3]/32); -- 5 times max pooling of 1/2
    local oW = math.floor(opt.sampleSize[4]/32); -- 5 times max pooling of 1/2

    model:add(nn.View(256*oT*oH*oW))                                                      -- 256 x [W/32] x [H/32] x [T/16] -> 256 * [W/32] * [H/32] * [T/16] 
    model:add(nn.Linear(256*oT*oH*oW, 2048))                                              -- 2048 -> 2048
    model:add(nn.Threshold(0,  0, true))
    model:add(nn.Dropout(opt.dropout))
    model:add(nn.Linear(2048, 2048))                                                       -- 2048 -> 2048
    model:add(nn.Threshold(0,  0, true))
    model:add(nn.Dropout(opt.dropout))
    model:add(nn.Linear(2048, nClasses))                                                  -- 2048 -> 101
    model:add(nn.LogSoftMax())

    model:cuda()
    return model
end
