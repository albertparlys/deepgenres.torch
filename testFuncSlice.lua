require 'image'
require 'paths'
require 'utils'
require 'torch'
local utils = paths.dofile('utils.lua')
local cfg = paths.dofile('config.lua')

function sliceFile(fileName,size)
	local status3, img = pcall(function() return image.load("'"..fileName.."'",1) end)
--	print(status3, img)	
	if not status3 then goto next end
	-- Compute approximate number of size x size samples
	local width = img:size()[2]
	local height = img:size()[1]
	local numSamples = math.floor(width/size)-1
	print(width,height,numSamples)
	-- Create directory to hold slices if there isn't one already
	local slicePath = cfg.dir.uji
--	if not paths.dir(slicePath) then
----		status = pcall(function () return paths.mkdir(slicePath) end)
--		if not status then print("Error making slice path for "..genre.." music slices") return 0 end
--	end
	for i=1,numSamples do
		print('Creating slice: '..i..'/'..numSamples..' for '..fn)
		local start = (i*size)+1
		local status,slice = pcall(function() return image.crop(img,start,1,start+size,size+1)end)
		if not status then goto next end
		local ext = paths.extname(fn)
		local base_fn = paths.basename(fn, ext)
		local numExt = utils.splitString(base_fn,"_")[2]
		local num = utils.splitString(numExt,".png")[1]
		local slice_fn = slicePath.."uji".."_"..""..i..".png"
		image.save(slice_fn, slice)  -- TODO add a check here
		currentNum = currentNum + 1
	end
	::next::

	
end
