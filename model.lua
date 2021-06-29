
function loadModel(pkg, numClasses, imgSize, phase)
  backend = pkg
  local batch
  if phase == 'train' then
    batch = imgSize
  else
    batch = 1
  end


  -- Input of size: batch X imgSize X imgSize
  -- Torch automatically infers batch size
  model = nn.Sequential()
  model:add(backend.SpatialConvolution(batch,64,3,3,1,1,1,1))
  model:add(nn.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(backend.SpatialConvolution(64,128,3,3,1,1,1,1))
  model:add(nn.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(backend.SpatialConvolution(128,256,3,3,1,1,1,1))
  model:add(nn.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(backend.SpatialConvolution(256,512,3,3,1,1,1,1))
  model:add(nn.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(nn.View(-1,512))
-- model:add(backend.Dropout(0.5))
  model:add(nn.Linear(512,1024))
  model:add(nn.ELU())
  model:add(nn.View(1024*8*8))
  model:add(nn.Linear(1024*8*8,numClasses))
  model:add(backend.LogSoftMax())
--  if phase == 'train' then
    --model:training()
--  else if phase == 'test' then
    --model:evaluate()
--  else
--    print('wrong state')
--  end
  	  if backend == 'nn' then

	  else
          cudnn.convert(model, cudnn)
	 -- model:cuda()
          end
	  --print(net)
  return model
end

return loadModel
