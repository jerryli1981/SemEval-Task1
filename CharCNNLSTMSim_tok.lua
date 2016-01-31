local CharCNNLSTMSim_tok = torch.class('CharCNNLSTMSim_tok')

function CharCNNLSTMSim_tok:__init(config)

  self.learning_rate = config.learning_rate or 0.01
  self.batch_size    = config.batch_size    or 25
  self.sim_nhidden   = config.sim_nhidden   or 50
  self.reg           = config.reg           or 1e-4

  self.num_layers    = config.num_layers    or 1

  self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

  self.dict = {}
  for i = 1,#self.alphabet do
    self.dict[self.alphabet:sub(i,i)] = i
  end

  self.char_vocab_size = #self.alphabet+1

  --if tok2char
  --self.inputFrameSize = 50

  --if tok2vec
  self.inputFrameSize = #self.alphabet

  self.max_sent_length = 10

  self.tok_length = 12

  self.outputFrameSize = 100

  self.emb_dim  = (self.outputFrameSize * 3)

  self.mem_dim = 100

  self.num_classes = 5

  self.kw = 3
  self.kw2 = 3

  self.pool_kw = 2
  self.pool_dw = 1

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate, momentum = 0.9, decay = 1e-5 }
  --self.optim_state = { learningRate = self.learning_rate }

  self.criterion = localize(nn.DistKLDivCriterion())

  self.sim_module = self:new_sim_module()

  self.params, self.grad_params = self.sim_module:getParameters()

end

function addCNNUnit(self, x)


  lookup_layer = nn.LookupTable(self.char_vocab_size, self.inputFrameSize)(x)

  conv_layer_1 = nn.ReLU()(nn.TemporalConvolution(self.inputFrameSize, self.outputFrameSize, self.kw)(lookup_layer))

  pool_layer_1 = nn.TemporalMaxPooling(self.pool_kw, self.pool_dw)(conv_layer_1)

  conv_layer_2 = nn.ReLU()(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2)(pool_layer_1))

  pool_layer_2 = nn.TemporalMaxPooling(self.pool_kw, self.pool_dw)(conv_layer_2)

  conv_layer_3 = nn.ReLU()(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2)(conv_layer_2))

  conv_layer_4 = nn.ReLU()(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2)(conv_layer_3))

  return nn.Reshape(self.emb_dim)(conv_layer_4)

end

function addLSTMUnit(prev_h, prev_c, x, inputSize, hiddenSize)

  -- Input gate. Equation (7)
  i_gate = nn.Sigmoid()(nn.CAddTable()({
    nn.Linear(inputSize,hiddenSize)(x),
    nn.Linear(hiddenSize,hiddenSize)(prev_h),
    nn.Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  -- Forget gate. Equation (8)
  f_gate = nn.Sigmoid()(nn.CAddTable()({
    nn.Linear(inputSize,hiddenSize)(x),
    nn.Linear(hiddenSize,hiddenSize)(prev_h),
    nn.Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  -- New contribution to c. Right term in equation (9)
  learning = nn.Tanh()(nn.CAddTable()({
    nn.Linear(inputSize,hiddenSize)(x),
    nn.Linear(hiddenSize,hiddenSize)(prev_h)
  }))
  -- Memory cell. Equation (9)
  c = nn.CAddTable()({
    nn.CMulTable()({f_gate, prev_c}),
    nn.CMulTable()({i_gate, learning})
  })
  -- Output gate. Equation (10)
  o_gate = nn.Sigmoid()(nn.CAddTable()({
    nn.Linear(inputSize,hiddenSize)(x),
    nn.Linear(hiddenSize,hiddenSize)(prev_h),
    nn.Linear(hiddenSize,hiddenSize)(c)
  }))
  -- Updated hidden state. Equation (11)
  h = nn.CMulTable()({o_gate, nn.Tanh()(c)})
  return h, c
end

function CharCNNLSTMSim_tok:new_sim_module()

  local inputs = {}

  local l_prev_h, l_prev_c = nn.Identity()(),nn.Identity()()
  local r_prev_h, r_prev_c = nn.Identity()(),nn.Identity()()

  table.insert(inputs, l_prev_h)
  table.insert(inputs, r_prev_h)

  table.insert(inputs, l_prev_c)
  table.insert(inputs, r_prev_c)

  for l = 1, 2*self.max_sent_length do

    local tok = nn.Identity()()
    table.insert(inputs, tok)

    local cnn_out = addCNNUnit(self, tok)


    if l <= self.max_sent_length then
      --lvec = nn.Tanh()( nn.Linear(self.emb_dim+ self.mem_dim, self.mem_dim)( nn.JoinTable(1)({lvec, cnn_out}) ) )
      l_prev_h, l_prev_c = addLSTMUnit(l_prev_h, l_prev_c, cnn_out, self.emb_dim, self.mem_dim)
    else
      --rvec = nn.Tanh()( nn.Linear(self.emb_dim+ self.mem_dim, self.mem_dim)( nn.JoinTable(1)({rvec, cnn_out}) ) )
      r_prev_h, r_prev_c = addLSTMUnit(r_prev_h, r_prev_c, cnn_out, self.emb_dim, self.mem_dim)
    end
  end

  local mult_dist = nn.CMulTable(){l_prev_h, r_prev_h}
  local add_dist = nn.Abs()(nn.CSubTable(){l_prev_h, r_prev_h})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  vecs_to_inputs = nn.gModule(inputs, {vec_dist_feats})

  local sim_module = nn.Sequential()
    :add(vecs_to_inputs)
    :add(nn.Linear(self.mem_dim*2, self.sim_nhidden))
    :add(nn.Sigmoid())   -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())

  return localize(sim_module)


end

function CharCNNLSTMSim_tok:train(dataset)


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

      --if using past data then use below code
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

        local inputs = {}

        local lvec = localize(torch.rand(self.mem_dim))
        local rvec = localize(torch.rand(self.mem_dim))

        table.insert(inputs, lvec)
        table.insert(inputs, rvec)

        local lc = localize(torch.rand(self.mem_dim))
        local rc = localize(torch.rand(self.mem_dim))

        table.insert(inputs, lc)
        table.insert(inputs, rc)

        for k = 1, self.max_sent_length do
          if k <= #lsent then
            tok = lsent[k]
            tok_vec = self:tok2characterIdx(tok)
            --tok_vec = self:tok2vec(tok)
            table.insert(inputs, tok_vec)
          else
            tok_vec = torch.Tensor(self.tok_length):fill(#self.alphabet+1)
            --tok_vec = localize(torch.rand(self.tok_length, #self.alphabet))
            table.insert(inputs, tok_vec)
          end
        end

        for k = 1, self.max_sent_length do
          if k <= #rsent then
            tok = rsent[k]
            tok_vec = self:tok2characterIdx(tok)
            --tok_vec = self:tok2vec(tok)
            table.insert(inputs, tok_vec)
          else
            tok_vec = torch.Tensor(self.tok_length):fill(#self.alphabet+1)
            --tok_vec = localize(torch.rand(self.tok_length, #self.alphabet))
            table.insert(inputs, tok_vec)
          end
        end

        local output = self.sim_module:forward(inputs)

        local example_loss = self.criterion:forward(output, targets[j])
        loss = loss + example_loss

        local sim_grad = self.criterion:backward(output, targets[j])

        self.sim_module:backward(inputs, sim_grad)

      end
      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      --loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      --self.grad_params:add(self.reg, self.params)

      avgloss = avgloss + loss
      return loss, self.grad_params
    end
    optim.sgd(feval, self.params, self.optim_state)
    avgloss = avgloss/N
  end
  xlua.progress(dataset.size, dataset.size)
  return avgloss
end

-- Predict the similarity of a sentence pair.
function CharCNNLSTMSim_tok:predict(lsent, rsent)

  local inputs = {}

  local lvec = localize(torch.rand(self.mem_dim))
  local rvec = localize(torch.rand(self.mem_dim))

  table.insert(inputs, lvec)
  table.insert(inputs, rvec)

  local lc = localize(torch.rand(self.mem_dim))
  local rc = localize(torch.rand(self.mem_dim))

  table.insert(inputs, lc)
  table.insert(inputs, rc)

  for k = 1, self.max_sent_length do
    if k <= #lsent then
      tok = lsent[k]
      tok_vec = self:tok2characterIdx(tok)
      --tok_vec = self:tok2vec(tok)
      table.insert(inputs, tok_vec)
    else
      tok_vec = torch.Tensor(self.tok_length):fill(#self.alphabet+1)
      --tok_vec = localize(torch.rand(self.tok_length, #self.alphabet))
      table.insert(inputs, tok_vec)
    end
  end

  for k = 1, self.max_sent_length do
    if k <= #rsent then
      tok = rsent[k]
      tok_vec = self:tok2characterIdx(tok)
      --tok_vec = self:tok2vec(tok)
      table.insert(inputs, tok_vec)
    else
      tok_vec = torch.Tensor(self.tok_length):fill(#self.alphabet+1)
      --tok_vec = localize(torch.rand(self.tok_length, #self.alphabet))
      table.insert(inputs, tok_vec)
    end
  end

  local output = self.sim_module:forward(inputs)

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

function CharCNNLSTMSim_tok:tok2characterIdx(token)

  local s = token:lower()
  local output = torch.Tensor(self.tok_length):fill(#self.alphabet+1)
  for i = #s, math.max(#s - self.tok_length + 1, 1), -1 do
    c = s:sub(i,i)
    if self.dict[c] then
      output[#s-i+1] = self.dict[c]
    else
      output[#s-i+1] = #self.alphabet+1
    end
  end
  return localize(output)
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
  return localize(t:transpose(1,2):contiguous())
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












