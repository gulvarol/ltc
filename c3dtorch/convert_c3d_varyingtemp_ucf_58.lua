require 'cudnn'
NOF = 20 -- 20, 40, 60, 80, 100
splitNo = 1 -- 1, 2, 3

print('Loading model.')
model=torch.load('c3d.t7')
lastlayer = torch.load('ucf' .. splitNo .. '_model11_rgb16_freeze_relu_dr05.t7')

print('Adjusting fc6 weights.')
W = lastlayer.modules[2].weight:double()
-- 4096 x 8192
print(W:size())   
U = torch.Tensor(W:size(1), W:size(2)/4)
-- 4096 x 2048
print(U:size())

for i = 1, W:size(1) do
   -- Take one row of input weight matrix (8192)
   rw = W[{{i}, {}}]:squeeze()
   -- Construct the row of output weight matrix (2048)
   ru = torch.Tensor(512*4)
   tempout = torch.Tensor(4)
   for j = 1, 512 do
      tempin = rw[{{16*(j-1)+1, 16*j}}]
      
      -- Sum 1,  2,  5,  6 * 16
      --     3,  4,  7,  8 * 16
      --     9, 10, 13, 14 * 16
      --    11, 12, 15, 16 * 16
      -- Sum the weights of top-left, top-right, bottom-left, bottom-right of 4x4 into 2x2
      --tempin=
      --  1  2  3  4
      --  5  6  7  8
      --  9 10 11 12
      -- 13 14 15 16
      
      --tempout=
      -- 1 2
      -- 3 4
      tempout[1] = tempin[1] + tempin[3] +  tempin[9] + tempin[11]
      tempout[2] = tempin[2] + tempin[4] + tempin[10] + tempin[12]
      tempout[3] = tempin[5] + tempin[7] + tempin[13] + tempin[15]
      tempout[4] = tempin[6] + tempin[8] + tempin[14] + tempin[16]

      ru[{{4*(j-1)+1, 4*j}}] = tempout
   end
   U[{{i}, {}}] = ru
end

model100 = nn.Sequential()
for i =1,21 do
   model100:add(model.modules[i])
end
if NOF == 100 then
   model100:add(nn.VolumetricMaxPooling(7, 1, 1, 7, 1, 1))
elseif NOF == 80 then
   model100:add(nn.VolumetricMaxPooling(5, 1, 1, 5, 1, 1))
elseif NOF == 60 then
   model100:add(nn.VolumetricMaxPooling(4, 1, 1, 4, 1, 1))
elseif NOF == 40 then
   model100:add(nn.VolumetricMaxPooling(3, 1, 1, 3, 1, 1))
elseif NOF ==20 then
   model100:add(nn.VolumetricMaxPooling(2, 1, 1, 2, 1, 1))
end

model100:add(nn.View(512*2*2))
model100:add(nn.Linear(2048, 4096))
model100.modules[24].weight = U
model100.modules[24].gradWeight = torch.Tensor(U:size())
--for i=24,29 do
--   model100:add(model.modules[i])
--end

model100:cuda()
model100 = cudnn.convert(model100, cudnn)

for i=3,12 do
   model100:add(lastlayer.modules[i])
end

print('Saving model.')

torch.save('ucf' .. splitNo .. '_c3d_' .. NOF ..'f_58.t7', model100)

print('Done.')
