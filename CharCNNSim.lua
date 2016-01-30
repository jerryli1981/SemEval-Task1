local CharCNNSim = torch.class('CharCNNSim')

function CharCNNSim:__init(config)

  self.learning_rate = config.learning_rate or 0.5
  self.batch_size    = config.batch_size    or 25
  self.sim_nhidden   = config.sim_nhidden   or 50

  --self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
  self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
  self.dict = {}
  for i = 1,#self.alphabet do
    self.dict[self.alphabet:sub(i,i)] = i
  end

  -- maxlen is 113
  self.length = 115

  self.inputFrameSize = #self.alphabet

  self.outputFrameSize = 128

  self.reshape_dim = 9 * self.outputFrameSize

  self.mem_dim = self.reshape_dim

  self.num_classes = 5

  self.kw = 7
  self.dw = 1
  self.kw2 = 3

  self.pool_kw = 3
  self.pool_dw = 3

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate, momentum = 0.9, decay = 1e-5 }

  self.criterion = localize(nn.DistKLDivCriterion())

  self.sim_module = self:new_sim_module()
  
  self.params, self.grad_params = self.sim_module:getParameters()

  share_params(self.sim_module, self.sim_module)

end

function CharCNNSim:new_sim_module()

  print("Call pure cnn sim module")

  local linput, rinput = nn.Identity()(), nn.Identity()()

  local lvec = nn.Linear(self.reshape_dim, 512)(
        nn.Reshape(self.reshape_dim)(
        nn.Threshold()(
        nn.TemporalConvolution(self.outputFrameSize,self.outputFrameSize,self.kw2, self.dw)(

        nn.TemporalMaxPooling(self.pool_kw, self.pool_dw)(
        nn.Threshold()(
        nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2, self.dw)(

        nn.TemporalMaxPooling(self.pool_kw, self.pool_dw)(
        nn.Threshold()(
        nn.TemporalConvolution(self.inputFrameSize, self.outputFrameSize, self.kw, self.dw)(linput))))))))))

  local rvec = nn.Linear(self.reshape_dim, 512)(
       
        nn.Reshape(self.reshape_dim)(
        nn.Threshold()(
        nn.TemporalConvolution(self.outputFrameSize,self.outputFrameSize,self.kw2, self.dw2)(

        nn.TemporalMaxPooling(self.pool_kw, self.pool_dw)(
        nn.Threshold()(
        nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2, self.dw)(

        nn.TemporalMaxPooling(self.pool_kw, self.pool_dw)(
        nn.Threshold()(
        nn.TemporalConvolution(self.inputFrameSize, self.outputFrameSize, self.kw, self.dw)(rinput))))))))))

  --local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  --local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  --local vec_dist_feats = nn.JoinTable(1){lvec, rvec}

  local vecs_to_input = nn.gModule({linput, rinput}, {add_dist})

  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(512, self.num_classes))
    --:add(nn.Tanh())   -- does better than tanh
    --:add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())

  return localize(sim_module)

end

function CharCNNSim:train(dataset)

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
        local linputs = self:seq2vec(lsent)
        local rinputs = self:seq2vec(rsent)
        inputs = {linputs, rinputs}
        local output = self.sim_module:forward(inputs)
        local example_loss = self.criterion:forward(output, targets[j])
        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        self.sim_module:backward(inputs, sim_grad)

      end
      loss = loss / batch_size
      self.grad_params:div(batch_size)

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
function CharCNNSim:predict(lsent, rsent)

  local linputs = self:seq2vec(lsent)
  local rinputs = self:seq2vec(rsent)
  inputs = {linputs, rinputs}
  local output = self.sim_module:forward(inputs)

  return localize(torch.range(1,5)):dot(output:exp())

end

-- Produce similarity predictions for each sentence pair in the dataset.
function CharCNNSim:predict_dataset(dataset)

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

function CharCNNSim:seq2vec(sequence)

  local s = ''
  --print(sequence)
  for i=1, #sequence do
    s = s .. sequence[i] .. " "
  end
  s = s:gsub("%s+", "")
  s = trim(s)

  local s = s:lower()
  local t = torch.Tensor(#self.alphabet, self.length)
  t:zero()
  for i = #s, math.max(#s - self.length + 1, 1), -1 do
    if self.dict[s:sub(i,i)] then
      t[self.dict[s:sub(i,i)]][#s - i + 1] = 1
    end
  end
  return localize(t:transpose(1,2))
end

function CharCNNSim:save(path)
  local config = {
    sim_nhidden = self.sim_nhidden
  }

  torch.save(path, {
    params = self.params,
    config = config,
    })

end

function CharCNNSim.load(path)
  local state = torch.load(path)
  local model = CharCNNSim.new(state.config)
  model.params:copy(state.params)
  return model
end












