require 'torch'
require 'nn'
require 'optim'
require 'paths'
require 'loader'
require 'xlua'
require 'pl'
require 'nnx'
require 'pretty-nn'
require 'gnuplot'
require 'cudnn'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-weights', true, 'Previously saved model weights to load')
cmd:option('-mode', 'train', 'Training mode')
cmd:option('-modelFactory', 'model.lua', 'Lua file to generate model definition')
cmd:option('-backend', 'cudnn', 'Set to cudnn to use GPU')
cmd:option('-logsTrainPath', './logs/training/', ' Path to save Training logs')
cmd:option('-logsValPath', './logs/val/', ' Path to save Validation logs')
cmd:option('-epochSave', true, 'save model every epoch')
cmd:option('-trainPath', './models/', ' Path to save model between epochs')
cmd:option('-saveName', 'deepgenre.t7', 'Name of serialized model')
cmd:option('-epochs', 1000, 'Number of epochs for training')
cmd:option('-learningRate', 0.01, 'Training learning rate')
cmd:option('-classes', 5, 'Number of genres to classify')
cmd:option('-o','adam','Optimizzation for CNN Learning')
cmd:option('-config', 'config.lua', 'Configuration file containing architecture params')
cmd:option('-batchSize', 1, 'Batch size in training')
cmd:option('-createSpectrograms', false, 'Create spectrograms and slice them if this has not already been done.')


cmd:text()
local opt = cmd:parse(arg)
print(opt)
local backend
if opt.backend == 'nn' then
backend = nn
else
backend = cudnn
end
local cfg = paths.dofile(opt.config)
local sliceAudio = paths.dofile('data.lua').sliceAudio
local rename = paths.dofile('data.lua').labelGenres
local trainTime = os.clock()

-- this matrix records the current confusion across classes
genres = (cfg.genres)
confusion = optim.ConfusionMatrix(genres)

	function train(net, lr, createSpectrograms)
	  if createSpectrograms then sliceAudio(cfg) end

	  print("Preparing dataset...")
	  local loader = Loader.new(model, cfg, 'train')
	  local X,y = loader:getDataset()
	  if opt.backend == 'nn' then
		backend = nn
	  else
		net:cuda()
		X:cuda()
		print(#X)
		y:cuda()
	  end
	  print("Dataset prepared!")
	  print("Training....")
	  parameters,gradParameters = net:getParameters()

	local criterion = nn.ClassNLLCriterion()
	criterion.sizeAverage = false
	criterion:cuda()
	if opt.o == 'sgd' then						
	local optimState = {learningRate = lr, learningRateDecay = 0.00001, momentum = 0.005}
	elseif opt.o == 'rmsprop' then
	local optimState = {learningRate = lr, alpha=0.9, epsilon=1e-8}
	elseif opt.o == 'adagrad' then
	local optimState = {learningRate = lr*0.1, learningRateDecay = 0, weightDecay=0}
	elseif opt.o == 'adadelta' then
	local optimState = {learningRate = lr*0.1, rho = 0.9, eps=0.000001, weightDecay = 0}
	elseif opt.o == 'adam' then
	local optimState = {learningRate = lr,learningRateDecay = 0.00001, epsilon=1e-8, beta1=0.9, beta2= 0.999, weightDecay=0.00001}
	else
	print('no optim detected')
	end

	  local startTime = os.time()
	  
	  print(net)

	  -- training
	  local predRep = optim.Logger(opt.logsTrainPath..'predTrain.log')
	  predRep:setNames{'Label','Output'}
	  local accuracyLog = optim.Logger(opt.logsTrainPath..'accuracy.log')
	  accuracyLog:setNames{'Global Error'}
	  for i=1, opt.epochs do
		    local timerEpoch = torch.Timer()
			    	for j = 1, X:size(1) do
						-- function to give to optim
						--print(X[j])		
						timerData= torch.Timer()
						feval = function(x)
						collectgarbage()
						--print(o:size())
						-- get new parameters
						if x ~= parameters then
							parameters:copy(x)
						end						
						--print(params:size())
						
						--reset gradients
						gradParameters:zero()

						--make inputs for net
						inputX = X[j]:clone()
						label = y[j] 
						input = inputX:view(1,X[j]:size(1),X[j]:size(2))
						input:cuda()
						--propagate
						output = net:forward(input)
						output:cuda()
						loss = criterion:forward(output,label)

						--learn
						gradOutput = criterion:backward(output,label)
						net:backward(input, gradOutput)


						--prediction
						desiredOutput = torch.exp(output)
						valuePredicted, predictedLabel = torch.sort(desiredOutput,true)
						-- update confusion
						for p = 1, opt.batchSize do
						confusion:add(predictedLabel[1], label)
						end
						predRep:add{label,predictedLabel[1]}

						--output
						return loss, gradParameters, output, gradInput, gradOutput		
			     	end
						-- optimize on current batch
						currentLoss = 0
						if opt.o == 'sgd' then						
						newX, fs = optim.sgd(feval, parameters, optimState)
						elseif opt.o == 'rmsprop' then
						newX, fs = optim.rmsprop(feval, parameters, optimState)
						elseif opt.o == 'adagrad' then
						newX, fs = optim.adagrad(feval, parameters, optimState)
						elseif opt.o == 'adam' then
						newX, fs = optim.adam(feval, parameters, optimState)
						elseif opt.o == 'adadelta' then
						newX, fs = optim.adam(feval, parameters, optimState)
						else
						print('no optim detected')
						end		
						xlua.progress(j, X:size(1))
					end
		print("\n================================="..i.."==============================================\n")
		print(confusion)
                local globalErr = 1 - (confusion.totalValid)
		accuracyLog:add{globalErr}
		accuracyLog:style{'+-'}
		accuracyLog:plot()
		confusion:zero()
		print("\n======================================================================================\n")
		-- save/log current net
		local filename = paths.concat(opt.logsValPath, 'deep.net')
		local modeldata = torch.save(opt.trainPath..opt.o..'_epochmodel_150_'..i..'.t7', net)
	end
print(string.format("Training Time: %.2f\n",os.clock() - trainTime))
torch.save(opt.trainPath..opt.saveName,net)
end

if opt.mode =='train' then
local modelFactory = paths.dofile(opt.modelFactory)
local model = modelFactory(backend, opt.classes, opt.batchSize, opt.mode)
train(model, opt.learningRate, opt.createSpectrograms)
testing = paths.dofile('test.lua')
test()
else if opt.mode == 'test' then
print("Better to use th test.lua")
testing = paths.dofile('test.lua')
test()
else
print('wrong state')
end
end
