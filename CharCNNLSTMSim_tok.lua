local CharCNNLSTMSim_tok = torch.class('CharCNNLSTMSim_tok')

function CharCNNLSTMSim_tok:__init(config)

  self.learning_rate = config.learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.sim_nhidden   = config.sim_nhidden   or 50
  self.reg           = config.reg           or 1e-4

  self.num_layers    = config.num_layers    or 1

  self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
  self.dict = {}
  for i = 1,#self.alphabet do
    self.dict[self.alphabet:sub(i,i)] = i
  end

  self.tok_length = 5

  self.outputFrameSize = 100

  self.reshape_dim = 3 * self.outputFrameSize

  self.emb_dim  = self.reshape_dim

  self.mem_dim = 100

  self.num_classes = 5

  -- optimizer configuration
 -- self.optim_state = { learningRate = self.learning_rate, momentum = 0.9, decay = 1e-5 }
  self.optim_state = { learningRate = self.learning_rate }

  self.criterion = localize(nn.DistKLDivCriterion())

  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
  }

  -- initialize cnn model
  local cnn_config = {
    seq_length = self.tok_length,
    inputFrameSize = #self.alphabet,
    outputFrameSize = self.outputFrameSize,
    reshape_dim = self.reshape_dim
  }

  self.tok_CNN = CharCNN(cnn_config)
  self.llstm = LSTM(lstm_config)
  self.rlstm = LSTM(lstm_config)

  self.sim_module = self:new_sim_module()

  local modules = nn.Parallel()
    :add(self.tok_CNN)
    :add(llstm)
    :add(self.sim_module)
  
  self.params, self.grad_params = modules:getParameters()

  share_params(self.tok_CNN, self.tok_CNN)
  share_params(self.rlstm, self.llstm)

end

function CharCNNLSTMSim_tok:new_sim_module()
  print('Using charCNNLSTMSim tok module')
  local vecs_to_input
  local lvec, rvec = nn.Identity()(), nn.Identity()()
  
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  local vecs_to_input = nn.gModule({lvec, rvec}, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(self.mem_dim*2, self.sim_nhidden))
    :add(nn.Sigmoid())   -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return localize(sim_module)
end

function CharCNNLSTMSim_tok:train(dataset)

  self.tok_CNN:training()
  self.llstm:training()
  self.rlstm:training()

  local indices = localize(torch.randperm(dataset.size))
  local avgloss = 0.
  local N = dataset.size / self.batch_size

  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local targets = localize(torch.zeros(batch_size, self.num_classes))
    for j=1, batch_size do
      local sim = dataset.sim_labels[indices[i+j-1]] * (self.num_classes-1)+1
      local ceil, floor = math.ceil(sim), math.floor(sim)

      if ceil ==0 then
        ceil=1
      end
      if floor ==0 then
        floor=1
      end

      if ceil == floor then
        targets[{j, floor}] = 1
      else
        targets[{j, floor}] = ceil -sim
        targets[{j, ceil}] = sim - floor
      end
    end

    local feval = function(x)
      self.grad_params:zero()

      local loss = 0
      for j = 1, batch_size do

        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]

        local linputs = {}
        for k = 1, #lsent do
          tok = lsent[k]
          tok_vec = self:tok2vec(tok)
          tok_cnn_output = self.tok_CNN:forward(tok_vec)
          table.insert(linputs, tok_cnn_output)
        end
        linputs = localize(nn.Reshape(#lsent, self.reshape_dim):forward(nn.JoinTable(1):forward(linputs)))
       
        local rinputs = {}
        for k = 1, #rsent do
          tok = rsent[k]
          tok_vec = self:tok2vec(tok)
          tok_cnn_output = self.tok_CNN:forward(tok_vec)
          table.insert(rinputs, tok_cnn_output)
        end

        rinputs = localize(nn.Reshape(#rsent, self.reshape_dim):forward(nn.JoinTable(1):forward(rinputs)))
       
        inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}

        local output = self.sim_module:forward(inputs)

        local example_loss = self.criterion:forward(output, targets[j])
        loss = loss + example_loss

        local sim_grad = self.criterion:backward(output, targets[j])

        local rep_grad = self.sim_module:backward(inputs, sim_grad)
        self:LSTM_CNN_backward(lsent, rsent, linputs, rinputs, rep_grad)

      end
      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2

      avgloss = avgloss + loss
      return loss, self.grad_params
    end
    optim.adagrad(feval, self.params, self.optim_state)
    avgloss = avgloss/N
  end
  xlua.progress(dataset.size, dataset.size)
  return avgloss
end

-- LSTM CNN backward propagation
function CharCNNLSTMSim_tok:LSTM_CNN_backward(lsent, rsent, linputs, rinputs, rep_grad)
  local lgrad, rgrad
  lgrad = localize(torch.zeros(#lsent, self.mem_dim))
  rgrad = localize(torch.zeros(#rsent, self.mem_dim))
  lgrad[#lsent] = rep_grad[1]
  rgrad[#rsent] = rep_grad[2]
  left = self.llstm:backward(linputs, lgrad)

  --[[
  for k = 1, #lsent do
    tok = lsent[k]
    tok_vec = self:tok2vec(tok)
    tok_vec = tok_vec:transpose(1,2):contiguous()
    vec = nn.NarrowTable(k):forward(left)[1]
    self.tok_CNN:backward(tok_vec, vec)
  end
  --]]

  right = self.rlstm:backward(rinputs, rgrad)

  --[[
  for k = 1, #rsent do
    tok = rsent[k]
    tok_vec = self:tok2vec(tok)
    tok_vec = tok_vec:transpose(1,2):contiguous()
    vec = nn.NarrowTable(k):forward(right)[1]
    self.tok_CNN:backward(tok_vec, vec)
  end
  --]]
  
end

-- Predict the similarity of a sentence pair.
function CharCNNLSTMSim_tok:predict(lsent, rsent)

  self.tok_CNN:evaluate()
  self.llstm:evaluate()
  self.rlstm:evaluate()

  local linputs = {}
  for k = 1, #lsent do
    tok = lsent[k]
    local tok_vec = self:tok2vec(tok)
    tok_cnn_output = self.tok_CNN:forward(tok_vec)
    table.insert(linputs, tok_cnn_output)
  end
  linputs = localize(nn.Reshape(#lsent, self.reshape_dim):forward(nn.JoinTable(1):forward(linputs)))


  local rinputs = {}
  for k = 1, #rsent do
    tok = rsent[k]
    local tok_vec = self:tok2vec(tok)
    tok_cnn_output = self.tok_CNN:forward(tok_vec)
    table.insert(rinputs, tok_cnn_output)
  end

  rinputs = localize(nn.Reshape(#rsent, self.reshape_dim):forward(nn.JoinTable(1):forward(rinputs)))


  inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
  
  local output = self.sim_module:forward(inputs)

  self.tok_CNN:forget()
  self.llstm:forget()
  self.rlstm:forget()

  return localize(torch.range(1,5)):dot(output:exp())

end


-- Produce similarity predictions for each sentence pair in the dataset.
function CharCNNLSTMSim_tok:predict_dataset(dataset)

  local predictions = localize(torch.Tensor(dataset.size))
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(lsent, rsent)
  end
  return predictions
end

function trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function CharCNNLSTMSim_tok:tok2vec(token)

  local s = token:lower()
  
  local t = torch.Tensor(#self.alphabet, self.tok_length)
  t:zero()
  for i = #s, math.max(#s - self.tok_length + 1, 1), -1 do
    if self.dict[s:sub(i,i)] then
      t[self.dict[s:sub(i,i)]][#s - i + 1] = 1
    end
  end
  return localize(t:transpose(1,2))
end

function CharCNNLSTMSim_tok:save(path)
  local config = {
    sim_nhidden = self.sim_nhidden
  }

  torch.save(path, {
    params = self.params,
    config = config,
    })

end

function CharCNNLSTMSim_tok.load(path)
  local state = torch.load(path)
  local model = CharCNNLSTMSim_tok.new(state.config)
  model.params:copy(state.params)
  return model
end












