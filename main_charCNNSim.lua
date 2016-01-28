require 'init' 

function pearson(x,y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -e,--epochs (default 10)        Number of training epochs
  -h,--sim_nhidden (default 50)   Number of sim_hidden
  -g,--debug  (default nil)       Debug setting  
  -o,--load (default nil)         Using previous model
  -c,--use_cuda (default -1)     Using cuda

]]

if args.use_cuda >= 0 then
  print("Using CUDA on GPU")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(args.use_cuda +1 )
else
  print("Using CPU")
  torch.setnumthreads(4)
end

localize = function(thing)
  if args.use_cuda >= 0 then
    return thing:cuda()
  end
  return thing
end

if args.debug == 'dbg' then
	dbg = require('debugger')
end

local data_dir = 'data/'

print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'

local train_dataset = read_dataset_char(train_dir)
local dev_dataset = read_dataset_char(dev_dir)
local test_dataset = read_dataset_char(test_dir)

printf('num train = %d\n', train_dataset.size)
printf('num dev = %d\n', dev_dataset.size)
printf('num test = %d\n', test_dataset.size)

-- get paths
local file_idx = 1
local model_save_path, model_save_pre_path

while true do
  model_save_path = string.format(
    models_dir .. '/charcnn-%d.th', file_idx)
  model_save_pre_path = string.format(
    models_dir .. '/charcnn-%d.th', file_idx-1)
  if lfs.attributes(model_save_path) == nil then
   break
  end
  file_idx = file_idx + 1
end


local model

model_class = CharCNNSim

if args.load == 'true' then
  print('using previous model ' .. model_save_pre_path)
  model = model_class.load(model_save_pre_path)
else
  print('initialize new model')

  model = model_class{
    sim_nhidden = args.sim_nhidden
    }  


end

-- number of epochs to train
local num_epochs = args.epochs

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  loss = model:train(train_dataset)
  printf('loss is: %.4f\n', loss)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_score = pearson(dev_predictions, localize(dev_dataset.sim_labels))

  printf('--dev score: %.4f\n', dev_score)

  if dev_score >= best_dev_score then
    best_dev_score = dev_score

    best_dev_model = model_class{
      sim_nhidden = args.sim_nhidden
      }
    

    best_dev_model.params:copy(model.params)

  end

end

printf('finished training in %.2fs\n', sys.clock() - train_start)


-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)
local test_score = pearson(test_predictions, localize(test_dataset.sim_labels))
printf('-- test score: %.4f\n', test_score)

--write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)


