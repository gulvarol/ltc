require 'image'
volmean = torch.load('data/volmean_sports1m.t7')
volmean = volmean[{{}, {}, {8, 119}, {30, 141}}] -- center crop 
w = 112
volmeanres = torch.Tensor(3, 16, w, w)
for t = 1, 16 do
   volmeanres[{{}, {t}, {}, {}}] = image.scale(volmean[{{}, {t}, {}, {}}]:squeeze(), w, w) -- resize
end
volmean = volmeanres:mean(2):squeeze()

print(volmean:size())
image.display(volmean)

torch.save('data/framemean_' .. w .. '_sports1m.t7', volmean)
