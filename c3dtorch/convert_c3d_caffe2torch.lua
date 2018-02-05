require 'loadcaffe'
require 'cudnn'

model = loadcaffe.load('c3d_sport1m_feature_extractor_video.prototxt', 'conv3d_deepnetA_sport1m_iter_1900000')
print(model)
model:remove(1)

model.modules[22] = nn.View(512*16)

model:apply(function(x)
   x.gradWeight = torch.CudaTensor()
   x.gradBias = torch.CudaTensor()
end)

model:cuda()
model = cudnn.convert(model, cudnn)

torch.save('c3d.t7', model)

-----------------------------------------------------------------------
volmean = torch.load('data/volmean_sports1m.t7')
volmean = volmean[{{}, {}, {1, 112}, {1, 112}}]:float()

input = torch.load('data/001_boxing_Ec2J9fKGDOs.mp4_14576_14676.t7')
input = input[{{}, {}, {1, 112}, {1, 112}}] -- take 112x112
input = input:index(1, torch.LongTensor{3, 2, 1}) -- bgr
output = model:forward(input:add(-volmean):cuda())

prob, class = torch.max(output, 1)
   
print(class)
