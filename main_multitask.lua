require 'init' 

function pearson(x,y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

function accuracy(pred,gold)
  return torch.eq(pred,gold):sum()/pred:size(1)
end

local args = lapp [[
Training script for semantic relatedness and entailment prediction on the SICK dataset.
  -d,--dim    (default 10)        LSTM memory dimension
  -e,--epochs (default 10)        Number of training epochs
  -l,--num_layers (default 1)     Number of layers
  -h,--sim_nhidden (default 50)   Number of sim_hidden
  -s,--structure (default lstm)   LSTM structure
  -g,--debug  (default nil)       Debug setting  
  -o,--load (default nil)         Using previous model

]]

if args.debug == 'dbg' then
	dbg = require('debugger')
end

local data_dir = 'data/'
local vocab = Vocab(data_dir .. 'vocab-cased.txt')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

local num_unk=0

local vecs = torch.Tensor(vocab.size, emb_dim)

for i = 1, vocab.size do
	local w = vocab:token(i)
	if emb_vocab:contains(w) then
		vecs[i] = emb_vecs[emb_vocab:index(w)]
	else
		num_unk = num_unk +1
		vecs[i]:uniform(-0.05, 0.05)
	end
end
print('unk count = ' ..num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = read_dataset(train_dir, vocab)
local dev_dataset = read_dataset(dev_dir, vocab)
local test_dataset = read_dataset(test_dir, vocab)

printf('num train = %d\n', train_dataset.size)
printf('num dev = %d\n', dev_dataset.size)
printf('num test = %d\n', test_dataset.size)

-- get paths
local file_idx = 1
local model_save_path, model_save_pre_path

while true do
  model_save_path = string.format(
    models_dir .. '/sim-ent-%s.%dl.%dd.%d.th', args.structure, args.num_layers, args.dim, file_idx)
  model_save_pre_path = string.format(
    models_dir .. '/sim-ent-%s.%dl.%dd.%d.th', args.structure, args.num_layers, args.dim, file_idx-1)
  if lfs.attributes(model_save_path) == nil then
   break
  end
  file_idx = file_idx + 1
end

local model_class = LSTM_Multitask_arc2

local model

if args.load == 'true' then
  print('using previous model ' .. model_save_pre_path)
  model = model_class.load(model_save_pre_path)
else
  print('initialize new model')

  model = model_class{
    emb_vecs   = vecs,
    mem_dim    = args.dim,
    structure = args.structure,
    num_layers = args.num_layers,
    sim_nhidden = args.sim_nhidden
    }  
end

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_sim_score = -1.0
local best_dev_ent_score = -1.0
local best_dev_model = model

header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  loss = model:train(train_dataset)
  printf('loss is: %.4f\n', loss)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  local dev_predictions= model:predict_dataset(dev_dataset)
  local dev_sim_score = pearson(dev_predictions[1], dev_dataset.sim_labels)
  local dev_ent_score = accuracy(dev_predictions[2], dev_dataset.ent_labels)

  printf('--dev sim score: %.4f\n', dev_sim_score)
  printf('--dev ent score: %.4f\n', dev_ent_score)

  if dev_sim_score >= best_dev_sim_score and dev_ent_score >= best_dev_ent_score then
    best_dev_sim_score = dev_sim_score
    best_dev_ent_score = dev_ent_score

    best_dev_model = model_class{
      emb_vecs   = vecs,
      mem_dim    = args.dim,
      structure = args.structure,
      num_layers = args.num_layers,
      sim_nhidden = args.sim_nhidden
      }
    
    best_dev_model.params:copy(model.params)

  end

end

printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev sim score = %.4f\n', best_dev_sim_score)
printf('-- using model with dev ent score = %.4f\n', best_dev_ent_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)

local test_sim_score = pearson(test_predictions[1], test_dataset.sim_labels)
local test_ent_score = accuracy(test_predictions[2], test_dataset.ent_labels)
printf('-- test sim score: %.4f\n', test_sim_score)
printf('-- test ent score: %.4f\n', test_ent_score)


if lfs.attributes(models_dir) == nil then
  lfs.mkdir(models_dir)
end


--write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)


