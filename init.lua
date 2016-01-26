require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('nnconv1d')
require('image')

include('read_data.lua')
include('Vocab.lua')
include('CRowAddTable.lua')
include('LSTM.lua')
include('LSTMSim.lua')
include('CharCNNSim.lua')
include('LSTMEnt.lua')
include('LSTM_Multitask_arc1.lua')
include('LSTM_Multitask_arc2.lua')


HighwayMLP = require 'HighwayMLP'

torch.setnumthreads(4)

printf = utils.printf

models_dir = 'trained_models'
predictions_dir = 'predictions'

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end
