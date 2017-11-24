require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   {check=function(paths)
       local out = true;
       for k,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             out = false
          end
       end
       return out
   end,
    name="paths",
    type="table",
    help="Multiple paths of directories with images"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

   -- find class names
   self.classes = {}
   local classPaths = {}
   if self.forceClasses then
      for k,v in pairs(self.forceClasses) do
         self.classes[k] = v
         classPaths[k] = {}
      end
   end
   local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   -- for each class,
   for k,path in ipairs(self.paths) do
      local dirs = dir.getdirectories(path);
      for k,dirpath in ipairs(dirs) do
         local class = paths.basename(dirpath)
         local idx = tableFind(self.classes, class)
         if not idx then
            table.insert(self.classes, class)
            idx = #self.classes
            classPaths[idx] = {}
         end
         if not tableFind(classPaths[idx], dirpath) then
            table.insert(classPaths[idx], dirpath);
         end
      end
   end

   self.classIndices = {}
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end

   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'avi'}

   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data

   print('running "find" on each class directory, and concatenate all'
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();

   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(self.classes) do
      -- iterate over classPaths
      for j,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions
            .. ' >>"' .. classFindFiles[i] .. '" \n'
         tmphandle:write(command)
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   print('now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   --==========================================================================
   print('load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. combinedFindList .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. combinedFindList .. "' |"
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end

   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '"
                                              .. classFindFiles[i] .. "' |"
                                              .. cut .. " -f1 -d' '"))
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
   --==========================================================================

   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
         local list = self.classList[i]
         local count = self.classList[i]:size(1)
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

-- size(), size(class)
function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- getByClass
function dataset:getByClass(class)
   local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
   local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
   return self:sampleHookTrain(imgpath)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, labelTable)
   local data, labels
   local quantity = #labelTable
   assert(dataTable[1]:dim() == 4)
   data = torch.Tensor(quantity, opt.sampleSize[1], opt.sampleSize[2], opt.sampleSize[3], opt.sampleSize[4])
   labels = torch.LongTensor(quantity):fill(-1111)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      labels[i] = labelTable[i]
   end
   return data, labels
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local labelTable = {}
   for i=1,quantity do
      local class = torch.random(1, #self.classes)
      local out = self:getByClass(class)
      while (out == nil) do -- if the sample is nil for some reason sample another one
         out = self:getByClass(class)
      end
      table.insert(dataTable, out)
      table.insert(labelTable, class)
   end
   local data, labels = tableToOutput(self, dataTable, labelTable)
   return data, labels
end

-- sampler, samples from the test set.
function dataset:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local labelTable = {}
   local indexTable = {}

   for i=1,quantity do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))  
      local out
      if(opt.crops10) then
        -- sample 4 corners + the center, and their flipped versions (in total 10 per test instance)
        for j=1,5 do

          out = self:sampleHookTest(imgpath, j, false)
          if(out == nil) then -- fill with zeros if the sample is nil
            table.insert(dataTable, torch.Tensor(opt.sampleSize[1], opt.sampleSize[2], opt.sampleSize[3], opt.sampleSize[4]):zero()) 
          else
            table.insert(dataTable, out)
          end
          table.insert(labelTable, self.imageClass[indices[i]])

          out = self:sampleHookTest(imgpath, j, true)
          if(out == nil) then
            table.insert(dataTable, torch.Tensor(opt.sampleSize[1], opt.sampleSize[2], opt.sampleSize[3], opt.sampleSize[4]):zero())
          else
            table.insert(dataTable, out)
          end
          table.insert(labelTable, self.imageClass[indices[i]])
        end
      else
      -- only center crop
        out = self:sampleHookTest(imgpath, 1, false)
        if(out == nil) then -- fill with zeros if the sample is nil
          table.insert(dataTable, torch.Tensor(opt.sampleSize[1], opt.sampleSize[2], opt.sampleSize[3], opt.sampleSize[4]):zero()) 
        else
          table.insert(dataTable, out)
        end
        table.insert(labelTable, self.imageClass[indices[i]])
      end
   end
   local data, labels = tableToOutput(self, dataTable, labelTable)
   return data, labels, indices
end

function dataset:computeAccuracy(scores)
   local N, M, P, S, X 
   -- Load dataLoader for videos ('self' dataLoader is for clips)
   local testCacheVid = torch.load(paths.concat(opt.logRoot, opt.dataset, 'cache', opt.stream, 'testCache.t7'))
   --assert(self.numSamples == scores:size(1))

   -- Find clip-video correspondence
   local clips = torch.Tensor(scores:size(1))
   for i = 1, scores:size(1) do
      N = ffi.string(torch.data(self.imagePath[i])) --v_ApplyEyeMakeup_g01_c01.avi_0017.avi
      N = paths.basename(N, 'avi')                  --v_ApplyEyeMakeup_g01_c01.avi_0017
      N = paths.basename(N, paths.extname(N))       --v_ApplyEyeMakeup_g01_c01
      
      for j = 1, testCacheVid.numSamples do
         if(N == paths.basename(ffi.string(torch.data(testCacheVid.imagePath[j])), 'avi')) then
            clips[i] = j
            break
         end
      end
   end

   -- Loop over videos to take mean over clip scores and compute accuracy
   local acc = 0
   for i = 1, testCacheVid.numSamples do
      X = clips:eq(i)                       -- 0,1 vector of clips belonging to same video
      if(X:any()) then                      -- normally always true
         X = torch.nonzero(X)              -- indices of clips belonging to same video
         S = scores:index(1,X:select(2,1)) -- scores of clips from the same video
         S = S:mean(1)                     -- mean over clips
         M, P = torch.max(S, 2)            -- max score -> predicted class
         if(P:squeeze() == testCacheVid.imageClass[i]) then -- pred==gt
            acc = acc + 1
         end
      end
   end
   return 100*acc/testCacheVid.numSamples
end

return dataset
