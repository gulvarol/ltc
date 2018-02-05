require 'torch'
require 'paths'

baseDir = '/home/gvarol/datasets/UCF101/flow/jpg/'
outputDir = '/home/gvarol/datasets/UCF101/flow/t7/'

function readJpgAsBinary(imfile)
    local fin = torch.DiskFile(imfile, 'r')
    fin:binary()
    fin:seekEnd()
    local file_size_bytes = fin:position() - 1
    fin:seek(1)
    local img_binary = torch.ByteTensor(file_size_bytes)
    fin:readByte(img_binary:storage())
    fin:close()
    return img_binary
end

for class in paths.iterdirs(baseDir) do
    print(class)
    for videoName in paths.iterdirs(paths.concat(baseDir,class)) do
        videoPath=paths.concat(baseDir,class,videoName)
        print(videoPath)
        videoTable={}
        count=0
        for image in paths.files(videoPath) do
            if(image:find('jpg$')) then
                count=count+1
            end
        end

        count = count/2

        videoTable.x={}
        videoTable.y={}
        for i=1,count do
            x_filename = paths.concat(videoPath, videoName..string.format('_%05d_x.jpg', i))
            y_filename = paths.concat(videoPath, videoName..string.format('_%05d_y.jpg', i))
            videoTable.x[i]=readJpgAsBinary(x_filename)
            videoTable.y[i]=readJpgAsBinary(y_filename)
        end

        os.execute('mkdir -p '..paths.concat(outputDir, class))
        outPath = paths.concat(outputDir, class)
        outFile = paths.concat(outputDir, class, videoName..'.t7')
        minmaxfile = paths.concat(baseDir,class,videoName, videoName..'_minmax.txt')
        os.execute('cp "'..minmaxfile..'" '..outPath)
        torch.save(outFile, videoTable)
    end
end

